//! Linux DMA-BUF import.
//!
//! A DMA-BUF is a kernel handle to buffer memory another process rendered
//! into. Importing one into `wgpu` needs the platform's own external-memory
//! path, and which one that is depends on the backend the `wgpu` device runs
//! on: Vulkan's `VK_EXT_external_memory_dma_buf` or EGL's
//! `EGL_LINUX_DMA_BUF_EXT`. [`DmaBufImporter`] picks between them once, from
//! the adapter, and hides the difference behind two operations:
//!
//! - [`DmaBufImporter::copy_to_texture`] for producers whose buffer is only
//!   valid for the duration of a callback: the copy is complete on the GPU
//!   before it returns.
//! - [`DmaBufImporter::copy_into`] for a render loop that already has a
//!   destination texture and can defer the buffer's release until its own
//!   submission completes.
//!
//! Both copy the frame's [visible extent](DmaBufFrame::visible_size) rather
//! than the whole allocation, because a producer may pad its buffer and
//! presenting the padding stretches the picture and draws the gutter.

mod frame;
mod gles;
mod vulkan;

pub use frame::{DRM_FORMAT_MOD_INVALID, DmaBufFormat, DmaBufFrame, DmaBufLease, DmaBufPlane};

use gles::GlesInterop;
use vulkan::ImportedVulkanImage;

#[derive(Debug)]
enum Backend {
    Vulkan,
    Gles(Box<GlesInterop>),
}

fn create_backend(backend: wgpu::Backend, device: &wgpu::Device) -> Backend {
    match backend {
        wgpu::Backend::Vulkan => Backend::Vulkan,
        wgpu::Backend::Gl => Backend::Gles(Box::new(GlesInterop::new(device))),
        backend => {
            panic!("DMA-BUF import requires a Vulkan or EGL/GLES wgpu device, received {backend:?}")
        }
    }
}

/// Whatever a deferred import allocated, alive until the GPU is done with it.
///
/// Drop this only once the submission that reads the imported frame has
/// completed — from `wgpu::Queue::on_submitted_work_done`, or after an explicit
/// `wgpu::Device::poll` on that submission.
#[derive(Debug)]
pub struct DmaBufImportGuard {
    #[expect(
        dead_code,
        reason = "the import is held only so that dropping the guard destroys it, which dead-code analysis does not count as a read"
    )]
    imported: Option<ImportedVulkanImage>,
}

/// The result of [`DmaBufImporter::copy_into`].
#[derive(Debug)]
pub struct DmaBufImport {
    /// The command buffers performing the copy, in submission order.
    ///
    /// `wgpu` forbids mixing raw and `wgpu` commands in one encoder, so the
    /// Vulkan path records the import into three: two raw ones holding the
    /// DMA-BUF queue-family acquire and release barriers, around a `wgpu`
    /// one holding the `copy_texture_to_texture` itself — which, recorded
    /// through `wgpu`, also marks the destination initialized for whatever
    /// uses it next. Submit them — ahead of [`Self::encoder`]'s finish — in a
    /// single `Queue::submit`, so the copy stays ordered before anything the
    /// caller records next. The EGL/GLES path performs its copy immediately
    /// and submits the encoder it was working with itself, so this is empty
    /// there.
    pub command_buffers: Vec<wgpu::CommandBuffer>,
    /// A fresh encoder for the caller's own `wgpu` commands.
    ///
    /// Record into it what should run after the copy, finish it, and submit
    /// it in the same `Queue::submit` as [`Self::command_buffers`], behind
    /// them.
    pub encoder: wgpu::CommandEncoder,
    /// Must outlive the submission of [`Self::command_buffers`] and
    /// [`Self::encoder`].
    pub guard: DmaBufImportGuard,
}

/// Imports Linux DMA-BUF frames into textures on one `wgpu` device.
#[derive(Debug)]
pub struct DmaBufImporter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: Backend,
}

impl DmaBufImporter {
    /// Creates an importer for `device`, resolving the platform import path
    /// from the adapter that device came from.
    ///
    /// # Panics
    ///
    /// Panics unless the adapter's backend is Vulkan or EGL/GLES; no other
    /// backend can import a DMA-BUF.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, adapter: &wgpu::Adapter) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            backend: create_backend(adapter.get_info().backend, device),
        }
    }

    /// Copies `frame` into a newly created texture owned exclusively by the
    /// caller, completing the copy on the GPU before returning.
    ///
    /// This is for producers whose buffer becomes invalid the moment their
    /// callback returns: the frame is presented and released here, and the
    /// returned texture no longer depends on it. Pixels are never read back to
    /// the CPU. The texture carries the frame's visible extent and is usable as
    /// `COPY_DST | TEXTURE_BINDING`.
    ///
    /// # Panics
    ///
    /// Panics when the frame's rendering fence has not signalled, or when GPU
    /// import, copying, or synchronization fails.
    #[must_use]
    pub fn copy_to_texture(&self, mut frame: DmaBufFrame) -> wgpu::Texture {
        assert!(
            frame.is_render_ready(),
            "a DMA-BUF must be ready before a synchronous GPU copy"
        );
        // The destination is the *visible* extent: a producer's buffer may be
        // allocated with alignment padding beyond it, and copying the padded
        // buffer then presenting it edge to edge stretches the picture and
        // draws the gutter. The source import still uses the buffer's own
        // dimensions and stride, so this only narrows what is taken from it.
        let (visible_width, visible_height) = frame.visible_size();
        let size = wgpu::Extent3d {
            width: visible_width,
            height: visible_height,
            depth_or_array_layers: 1,
        };
        let destination = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wgpu_external_frame_owned_dma_buf"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: frame.format.texture_format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let import = self.copy_into(&mut frame, &destination);
        // `command_buffers` carries the copy on Vulkan and is empty on GLES,
        // where the blit already ran on the calling thread. The follow-up
        // encoder is empty; finishing it keeps the submission one expression
        // for both backends.
        let submission = self.queue.submit(
            import
                .command_buffers
                .into_iter()
                .chain([import.encoder.finish()]),
        );
        match &self.backend {
            Backend::Vulkan => {
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(submission),
                        timeout: None,
                    })
                    .expect("the Vulkan DMA-BUF copy failed");
            }
            Backend::Gles(gles) => {
                // The GL blit ran outside any submission, so the poll above
                // cannot cover it; `glFinish` is what makes it complete
                // before the buffer goes back to the producer.
                gles.finish();
            }
        }
        frame.presented();
        drop(import.guard);
        frame.release(None);
        destination
    }

    /// Copies `frame`'s visible extent into `destination`, deferring the wait.
    ///
    /// `destination` must be a `COPY_DST` texture of the frame's
    /// [`DmaBufFormat::texture_format`] and at least its visible extent. The
    /// caller owns the rest of the protocol: call [`DmaBufFrame::presented`]
    /// once this returns, submit [`DmaBufImport::command_buffers`] followed
    /// by [`DmaBufImport::encoder`]'s finish in one `Queue::submit`, and only
    /// after that submission completes drop [`DmaBufImport::guard`] and call
    /// [`DmaBufFrame::release`].
    ///
    /// # Panics
    ///
    /// Panics when GPU import or copying fails.
    #[must_use]
    pub fn copy_into(&self, frame: &mut DmaBufFrame, destination: &wgpu::Texture) -> DmaBufImport {
        let (visible_width, visible_height) = frame.visible_size();
        let size = wgpu::Extent3d {
            width: visible_width,
            height: visible_height,
            depth_or_array_layers: 1,
        };
        let mut command_buffers = Vec::with_capacity(3);
        let imported = match &self.backend {
            Backend::Vulkan => {
                // The DMA-BUF queue-family barriers are raw commands, which
                // `wgpu` forbids in an encoder that also carries `wgpu`
                // commands. They get their own encoders around a `wgpu` one
                // holding `copy_texture_to_texture`; the three buffers are
                // returned for one submission, keeping the barriers
                // bracketing the copy and the copy's own transition and
                // initialization of `destination` intact.
                let imported = vulkan::import_dma_buf(&self.device, frame);
                let mut acquire =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("wgpu_external_frame_dma_buf_acquire"),
                        });
                imported.record_acquire(&mut acquire);
                let mut copy =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("wgpu_external_frame_dma_buf_copy"),
                        });
                copy.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: imported.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: destination,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    size,
                );
                let mut release =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("wgpu_external_frame_dma_buf_release"),
                        });
                imported.record_release(&mut release);
                command_buffers.extend([acquire.finish(), copy.finish(), release.finish()]);
                Some(imported)
            }
            Backend::Gles(gles) => {
                // `wgpu` tracks texture initialization lazily: a texture that
                // has never been written through `wgpu` counts uninitialized,
                // and the next `wgpu` command that reads or writes it records
                // a zero-clear at the head of that submission. The GL blit
                // below writes `destination` outside `wgpu`'s command stream,
                // so without a marker it would still count uninitialized and
                // such a lazy clear would erase the blit afterwards. A partial
                // `copy_buffer_to_texture` is the smallest write `wgpu`
                // accepts: it makes `wgpu` zero-initialize the whole mip level
                // and mark it initialized. Submitting it now also puts that
                // GPU work on the driver's queue ahead of the blit. `clear_texture`
                // cannot serve as the marker: it clears but never marks a
                // texture initialized, so the same lazy clear would still be
                // scheduled for the texture's next use.
                let texel_size = destination
                    .format()
                    .block_copy_size(None)
                    .expect("the DMA-BUF texture format must be block-copyable");
                let source = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wgpu_external_frame_dma_buf_init_marker"),
                    size: u64::from(texel_size),
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let mut marker =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("wgpu_external_frame_dma_buf_init"),
                        });
                marker.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &source,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: None,
                            rows_per_image: None,
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: destination,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                self.queue.submit([marker.finish()]);
                gles.copy_dma_buf(frame, destination);
                None
            }
        };
        DmaBufImport {
            command_buffers,
            encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_external_frame_dma_buf_follow_up"),
                }),
            guard: DmaBufImportGuard { imported },
        }
    }
}
