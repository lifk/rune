//! End-to-end tests of Vulkan and GLES DMA-BUF import.
//!
//! Each test allocates a DMA-BUF as a DRM dumb buffer on `vkms`, the virtual
//! KMS driver — real dumb-buffer ioctls and real PRIME export, no hardware —
//! writes a known pattern into it from the CPU, imports it on a `wgpu`
//! device, and checks the texels the GPU ends up holding.
//!
//! The tests assert the environment they need up front: a `vkms` card under
//! `/dev/dri`, an adapter for the backend under test, and the backend's own
//! DMA-BUF import support — the `VK_KHR_external_memory_fd`,
//! `VK_EXT_external_memory_dma_buf`, and `VK_EXT_image_drm_format_modifier`
//! device extensions on Vulkan, `EGL_EXT_image_dma_buf_import` on EGL/GLES.
//! Any of those missing fails the test rather than skipping it. On lavapipe
//! the Vulkan extension set is gated on `/dev/udmabuf` existing: without it
//! Mesa does not advertise `VK_EXT_image_drm_format_modifier`, so the
//! Vulkan tests need both devices passed through.
#![cfg(target_os = "linux")]

use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::sync::mpsc;

use wgpu_external_frame::dma_buf::{
    DmaBufFormat, DmaBufFrame, DmaBufImporter, DmaBufLease, DmaBufPlane,
};

/// `DRM_FORMAT_MOD_LINEAR` from `drm_fourcc.h`: the only layout a dumb
/// buffer has.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

/// Mirrors `struct drm_version` in `drm.h`. The pointers are buffers the
/// kernel writes into; the `_len` fields carry `__kernel_size_t`.
#[repr(C)]
struct DrmVersion {
    major: i32,
    minor: i32,
    patchlevel: i32,
    name_len: usize,
    name: *mut u8,
    date_len: usize,
    date: *mut u8,
    desc_len: usize,
    desc: *mut u8,
}

/// Mirrors `struct drm_mode_create_dumb` in `drm_mode.h`.
#[repr(C)]
struct DrmCreateDumb {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32,
    handle: u32,
    pitch: u32,
    size: u64,
}

/// Mirrors `struct drm_mode_map_dumb` in `drm_mode.h`.
#[repr(C)]
struct DrmMapDumb {
    handle: u32,
    pad: u32,
    offset: u64,
}

/// Mirrors `struct drm_mode_destroy_dumb` in `drm_mode.h`.
#[repr(C)]
struct DrmDestroyDumb {
    handle: u32,
}

/// Mirrors `struct drm_prime_handle` in `drm.h`.
#[repr(C)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

// The `drm.h` ioctl numbers, already encoded: `_IOW`/`_IOWR` with type 'd'.
const DRM_IOCTL_VERSION: libc::c_ulong = 0xc040_6400;
const DRM_IOCTL_PRIME_HANDLE_TO_FD: libc::c_ulong = 0xc00c_642d;
const DRM_IOCTL_MODE_CREATE_DUMB: libc::c_ulong = 0xc020_64b2;
const DRM_IOCTL_MODE_MAP_DUMB: libc::c_ulong = 0xc010_64b3;
const DRM_IOCTL_MODE_DESTROY_DUMB: libc::c_ulong = 0xc004_64b4;

/// `PRIME_HANDLE_TO_FD`'s `flags`: `DRM_RDWR | DRM_CLOEXEC` — read/write
/// access for the importer, close-on-exec so the descriptor cannot leak.
const PRIME_FD_FLAGS: u32 = (libc::O_RDWR | libc::O_CLOEXEC) as u32;

/// A CPU-writable DMA-BUF: a DRM dumb buffer allocated on `vkms`.
///
/// `vkms` is the kernel's virtual KMS driver — no hardware, but real
/// dumb-buffer ioctls and real PRIME export. Its buffers are linear and
/// CPU-mappable, so the same allocation takes the pattern write and the
/// import.
struct DumbBuf {
    /// The `vkms` card the buffer lives on; it stays open because every
    /// ioctl on `handle` goes through it.
    card: std::fs::File,
    /// The GEM handle `CREATE_DUMB` returned on `card`.
    handle: u32,
    /// The DMA-BUF descriptor exported from `handle`, duplicated into each
    /// frame built from this buffer.
    fd: OwnedFd,
    /// Row pitch in bytes, as `CREATE_DUMB` decided it.
    pitch: u32,
    /// Byte size of the allocation, as `CREATE_DUMB` decided it.
    size: usize,
}

impl DumbBuf {
    /// Allocates a `width`×`height` dumb buffer on the `vkms` card and
    /// exports it as a DMA-BUF.
    fn allocate(width: u32, height: u32) -> Self {
        let card = Self::open_vkms();
        let mut create = DrmCreateDumb {
            height,
            width,
            bpp: 32,
            flags: 0,
            handle: 0,
            pitch: 0,
            size: 0,
        };
        assert_eq!(
            // SAFETY: `CREATE_DUMB` writes the `drm_mode_create_dumb` passed
            // by pointer — `create` is a live local for the call — and fills
            // its `handle`/`pitch`/`size` outputs.
            unsafe { libc::ioctl(card.as_raw_fd(), DRM_IOCTL_MODE_CREATE_DUMB, &mut create) },
            0,
            "DRM_IOCTL_MODE_CREATE_DUMB failed: {}",
            io::Error::last_os_error()
        );
        let mut prime = DrmPrimeHandle {
            handle: create.handle,
            flags: PRIME_FD_FLAGS,
            fd: -1,
        };
        assert_eq!(
            // SAFETY: `PRIME_HANDLE_TO_FD` reads `prime.handle`, a live GEM
            // handle on `card`, and writes `prime.fd`, the descriptor it returns.
            unsafe { libc::ioctl(card.as_raw_fd(), DRM_IOCTL_PRIME_HANDLE_TO_FD, &mut prime) },
            0,
            "DRM_IOCTL_PRIME_HANDLE_TO_FD failed: {}",
            io::Error::last_os_error()
        );
        assert!(prime.fd >= 0, "PRIME_HANDLE_TO_FD returned no descriptor");
        Self {
            card,
            handle: create.handle,
            // SAFETY: `prime.fd` was just returned open by the ioctl.
            fd: unsafe { OwnedFd::from_raw_fd(prime.fd) },
            pitch: create.pitch,
            size: usize::try_from(create.size).expect("the buffer size fits usize"),
        }
    }

    /// Opens the `vkms` card under `/dev/dri`, identified by the driver's own
    /// name — not by a fixed node, since `card0` is not guaranteed to be it.
    fn open_vkms() -> std::fs::File {
        let entries = std::fs::read_dir("/dev/dri")
            .expect("allocating a DMA-BUF requires /dev/dri (the tests need `modprobe vkms`)");
        let mut card = None;
        for entry in entries {
            let path = entry.expect("reading /dev/dri failed").path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !name.starts_with("card") {
                continue;
            }
            let file = std::fs::File::options()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap_or_else(|error| panic!("opening {} failed: {error}", path.display()));
            if Self::driver_name(&file) == "vkms" {
                card = Some(file);
                break;
            }
        }
        card.expect("no vkms card under /dev/dri — the tests need vkms loaded")
    }

    /// The DRM driver's name for `card`, via `DRM_IOCTL_VERSION`.
    fn driver_name(card: &std::fs::File) -> String {
        let mut version = DrmVersion {
            major: 0,
            minor: 0,
            patchlevel: 0,
            name_len: 0,
            name: std::ptr::null_mut(),
            date_len: 0,
            date: std::ptr::null_mut(),
            desc_len: 0,
            desc: std::ptr::null_mut(),
        };
        assert_eq!(
            // SAFETY: `VERSION` reads `version`'s length fields and writes
            // back the lengths the buffers would need; no buffers are
            // passed, so nothing else is touched.
            unsafe { libc::ioctl(card.as_raw_fd(), DRM_IOCTL_VERSION, &mut version) },
            0,
            "DRM_IOCTL_VERSION failed: {}",
            io::Error::last_os_error()
        );
        let mut name = vec![0u8; version.name_len];
        version.name = name.as_mut_ptr();
        assert_eq!(
            // SAFETY: `version.name` points at `name_len` writable bytes, which
            // is exactly what the previous call asked for.
            unsafe { libc::ioctl(card.as_raw_fd(), DRM_IOCTL_VERSION, &mut version) },
            0,
            "DRM_IOCTL_VERSION failed: {}",
            io::Error::last_os_error()
        );
        String::from_utf8(name).expect("the DRM driver name is ASCII")
    }

    /// Writes `texel(x, y)` into every pixel of the buffer, at `self.pitch`
    /// bytes per row.
    fn fill(&self, texel: impl Fn(u32, u32) -> [u8; 4]) {
        let mut map_dumb = DrmMapDumb {
            handle: self.handle,
            pad: 0,
            offset: 0,
        };
        assert_eq!(
            // SAFETY: `MAP_DUMB` reads `map_dumb.handle`, a live GEM handle on
            // `card`, and writes `map_dumb.offset`, the `mmap` offset it returns.
            unsafe {
                libc::ioctl(
                    self.card.as_raw_fd(),
                    DRM_IOCTL_MODE_MAP_DUMB,
                    &mut map_dumb,
                )
            },
            0,
            "DRM_IOCTL_MODE_MAP_DUMB failed: {}",
            io::Error::last_os_error()
        );
        // SAFETY: `mmap` maps `self.size` bytes of `card` at the offset
        // `MAP_DUMB` returned, shared so the writes land in the buffer.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                self.size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.card.as_raw_fd(),
                libc::off_t::try_from(map_dumb.offset).expect("the map offset fits off_t"),
            )
        };
        assert_ne!(
            map,
            libc::MAP_FAILED,
            "mmap of the dumb buffer failed: {}",
            io::Error::last_os_error()
        );
        let stride = usize::try_from(self.pitch).expect("the pitch fits usize");
        for y in 0..usize::try_from(HEIGHT).expect("HEIGHT fits usize") {
            for x in 0..usize::try_from(WIDTH).expect("WIDTH fits usize") {
                let offset = y * stride + x * 4;
                assert!(offset + 4 <= self.size, "the pitch overflows the buffer");
                // SAFETY: `map` covers `self.size` bytes and `offset + 4`
                // stays within them, as the bounds just checked.
                unsafe {
                    map.cast::<u8>().add(offset).copy_from_nonoverlapping(
                        texel(
                            u32::try_from(x).expect("the coordinate fits u32"),
                            u32::try_from(y).expect("the coordinate fits u32"),
                        )
                        .as_ptr(),
                        4,
                    );
                }
            }
        }
        assert_eq!(
            // SAFETY: `map` is the live mapping of `self.size` bytes from
            // above, unmapped exactly once here after the writes it served
            // are done.
            unsafe { libc::munmap(map, self.size) },
            0,
            "munmap of the dumb buffer failed: {}",
            io::Error::last_os_error()
        );
    }

    /// A frame presenting the buffer as one linear RGBA plane.
    fn frame(&self, lease: Box<dyn DmaBufLease>) -> DmaBufFrame {
        // SAFETY: `self.fd` is open; `dup` returns a fresh descriptor or -1.
        let fd = unsafe { libc::dup(self.fd.as_raw_fd()) };
        assert!(
            fd >= 0,
            "dup of the DMA-BUF descriptor failed: {}",
            io::Error::last_os_error()
        );
        // SAFETY: `fd` was just returned open by `dup`.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        DmaBufFrame::new(
            WIDTH,
            HEIGHT,
            DmaBufFormat::Rgba8,
            DRM_FORMAT_MOD_LINEAR,
            vec![DmaBufPlane {
                fd,
                offset: 0,
                stride: self.pitch,
            }],
            None,
        )
        .with_lease(lease)
    }
}

impl Drop for DumbBuf {
    fn drop(&mut self) {
        let mut destroy = DrmDestroyDumb {
            handle: self.handle,
        };
        // SAFETY: `DESTROY_DUMB` reads `destroy.handle`, the live GEM handle
        // this buffer still owns on `card`. The exported DMA-BUF keeps its
        // own reference, so frames already built stay valid.
        unsafe {
            libc::ioctl(
                self.card.as_raw_fd(),
                DRM_IOCTL_MODE_DESTROY_DUMB,
                &mut destroy,
            );
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    importer: DmaBufImporter,
    pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    fn new(backends: wgpu::Backends) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: true,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .expect("the tests require an adapter");
        match adapter.get_info().backend {
            wgpu::Backend::Vulkan => assert_vulkan_import_supported(&adapter),
            wgpu::Backend::Gl => assert_gles_import_supported(&adapter),
            backend => panic!("the tests require a Vulkan or GLES adapter, received {backend:?}"),
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to open the device");
        let importer = DmaBufImporter::new(&device, &queue, &adapter);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dma_buf_read_texels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dma_buf.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dma_buf_read_texels"),
            layout: None,
            module: &module,
            entry_point: Some("read_texels"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            importer,
            pipeline,
        }
    }

    /// Records a dispatch sampling every texel of `view` into `readback`
    /// through the storage buffer `output`. `readback` must be `MAP_READ`
    /// with room for `width * height` packed texels.
    fn record_read_texels(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        readback: &wgpu::Buffer,
    ) {
        let size = u64::from(width * height) * 4;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dma_buf_texels"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dma_buf_texels"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, readback, 0, size);
    }

    /// Samples every texel of `view` (`width` × `height`) on the GPU and
    /// returns them as packed RGBA8 values, red in the low byte.
    fn read_texels(&self, view: &wgpu::TextureView, width: u32, height: u32) -> Vec<u32> {
        let readback = self.readback_buffer(u64::from(width * height) * 4);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record_read_texels(&mut encoder, view, width, height, &readback);
        let submission = self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("waiting for the GPU failed");
        self.map(&readback)
    }

    /// Maps `buffer` and reads it as packed little-endian texels.
    fn map(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let slice = buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("the map result receiver is alive");
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("waiting for the GPU failed");
        receiver
            .recv()
            .expect("the map callback ran")
            .expect("failed to map the readback buffer");
        let texels = slice
            .get_mapped_range()
            .expect("the readback buffer is mapped")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        buffer.unmap();
        texels
    }

    fn readback_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dma_buf_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LeaseEvent {
    Presented,
    Released,
}

#[derive(Debug)]
struct RecordingLease(mpsc::Sender<LeaseEvent>);

impl DmaBufLease for RecordingLease {
    fn presented(&mut self) {
        self.0
            .send(LeaseEvent::Presented)
            .expect("the lease event receiver is alive");
    }

    fn release(self: Box<Self>, _release_fence: Option<OwnedFd>) {
        self.0
            .send(LeaseEvent::Released)
            .expect("the lease event receiver is alive");
    }
}

/// Asserts the Vulkan environment the import needs: an adapter whose
/// physical device offers the three external-memory extensions `wgpu`
/// enables for `VK_EXT_external_memory_dma_buf` import.
fn assert_vulkan_import_supported(adapter: &wgpu::Adapter) {
    // The import's device extensions come from `wgpu`, which enables them
    // whenever the driver offers them; assert the offer so a device that
    // cannot run the tests fails here rather than inside the import.
    // SAFETY: `Adapter::as_hal` requires naming the adapter's real
    // backend; a non-Vulkan adapter yields `None` and panics. The guard
    // is only read.
    let hal_adapter = unsafe {
        adapter
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("the Vulkan tests require a Vulkan adapter")
    };
    let capabilities = hal_adapter.physical_device_capabilities();
    for extension in [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
        ash::ext::image_drm_format_modifier::NAME,
    ] {
        assert!(
            capabilities.supports_extension(extension),
            "the Vulkan adapter cannot import DMA-BUFs: {extension:?} is not supported"
        );
    }
}

/// Asserts the GLES environment the import needs: a `Gl` adapter whose EGL
/// display offers `EGL_EXT_image_dma_buf_import`.
fn assert_gles_import_supported(adapter: &wgpu::Adapter) {
    type EglGetCurrentDisplay = unsafe extern "C" fn() -> *mut core::ffi::c_void;
    type EglQueryString = unsafe extern "C" fn(*mut core::ffi::c_void, i32) -> *const i8;
    const EGL_EXTENSIONS: i32 = 0x3055;
    // SAFETY: `Library::new` is unsafe because `dlopen` runs the library's
    // initializers; this is the system EGL runtime by its versioned SONAME,
    // which the GLES adapter already has loaded, so the handle is the
    // existing one and no new initializer runs.
    let egl = unsafe { libloading::Library::new("libEGL.so.1") }
        .expect("the GLES tests require libEGL.so.1");
    // SAFETY: the signatures are EGL's own for these names, and the resolved
    // pointers stay valid while `egl` is alive, which is this function.
    let get_current_display = unsafe {
        *egl.get::<EglGetCurrentDisplay>(b"eglGetCurrentDisplay\0")
            .expect("libEGL.so.1 must export eglGetCurrentDisplay")
    };
    // SAFETY: as above, for `eglQueryString`.
    let query_string = unsafe {
        *egl.get::<EglQueryString>(b"eglQueryString\0")
            .expect("libEGL.so.1 must export eglQueryString")
    };
    // `wgpu` binds its EGL context only while one of its own calls runs, so
    // take the adapter's context lock while reading `eglGetCurrentDisplay`
    // and `eglQueryString`.
    // SAFETY: `Adapter::as_hal` requires the adapter's real backend; a
    // non-GLES adapter yields `None` and panics. The guard is only read.
    let hal_adapter = unsafe {
        adapter
            .as_hal::<wgpu::hal::api::Gles>()
            .expect("the GLES tests require a GLES adapter")
    };
    let _context = hal_adapter.adapter_context().lock();
    // SAFETY: `get_current_display` reads this thread's current EGL binding,
    // which the lock above just made the adapter's; `display` is then a live
    // `EGLDisplay` for `query_string`, and the returned C string outlives the
    // read while the display stays open, which the adapter guarantees.
    let extensions = unsafe {
        let display = get_current_display();
        assert!(
            !display.is_null(),
            "the GLES adapter's EGL context must be current under its own lock"
        );
        core::ffi::CStr::from_ptr(query_string(display, EGL_EXTENSIONS))
            .to_str()
            .expect("the EGL extension string is ASCII")
            .to_owned()
    };
    assert!(
        extensions
            .split(' ')
            .any(|extension| extension == "EGL_EXT_image_dma_buf_import"),
        "the EGL display cannot import DMA-BUFs: EGL_EXT_image_dma_buf_import is missing"
    );
}

/// The pattern written into the buffer: a texel that differs between
/// neighbouring pixels, rows, and channels, so a wrong row pitch or channel
/// order cannot reproduce it.
fn texel(x: u32, y: u32) -> [u8; 4] {
    [
        u8::try_from(x * 4).expect("fits"),
        u8::try_from(y * 5).expect("fits"),
        u8::try_from((x + y) * 2).expect("fits"),
        u8::try_from(255 - x).expect("fits"),
    ]
}

fn expected_texels() -> Vec<u32> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| u32::from_le_bytes(texel(x, y))))
        .collect()
}

/// `copy_to_texture` must import the buffer's contents: a shader reading the
/// returned texture sees exactly the pattern the CPU wrote.
fn copy_to_texture_imports_the_frame_on(backends: wgpu::Backends) {
    let gpu = Gpu::new(backends);
    let buffer = DumbBuf::allocate(WIDTH, HEIGHT);
    buffer.fill(texel);
    let (events, received) = mpsc::channel();
    let texture = gpu
        .importer
        .copy_to_texture(buffer.frame(Box::new(RecordingLease(events))));
    assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released],
        "a synchronous copy presents and releases the frame before returning"
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampled = gpu.read_texels(&view, WIDTH, HEIGHT);
    assert_eq!(sampled, expected_texels(), "sampled texels");
}
/// `copy_into` defers the wait: the caller presents the frame, records its own
/// commands into the returned encoder, and submits it in one `Queue::submit`
/// behind the import's command buffers. The readback dispatch here is the
/// caller's stand-in for a frame's real work — if the import's buffers and
/// the caller's were wrongly mixed or misordered, this either panics at
/// encode time or reads back an uninitialized texture instead of the frame.
/// `expected_command_buffers` is 3 on Vulkan — acquire, copy, release — and
/// 0 on GLES, which copies at `copy_into` call time instead.
fn copy_into_orders_the_copy_before_caller_commands_on(
    backends: wgpu::Backends,
    expected_command_buffers: usize,
) {
    let gpu = Gpu::new(backends);
    let buffer = DumbBuf::allocate(WIDTH, HEIGHT);
    buffer.fill(texel);
    let (events, received) = mpsc::channel();
    let destination = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dma_buf_copy_into_destination"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let mut frame = buffer.frame(Box::new(RecordingLease(events)));
    let mut import = gpu.importer.copy_into(&mut frame, &destination);
    assert_eq!(
        import.command_buffers.len(),
        expected_command_buffers,
        "unexpected number of import command buffers"
    );
    frame.presented();
    let readback = gpu.readback_buffer(u64::from(WIDTH * HEIGHT) * 4);
    let view = destination.create_view(&wgpu::TextureViewDescriptor::default());
    gpu.record_read_texels(&mut import.encoder, &view, WIDTH, HEIGHT, &readback);
    let submission = gpu.queue.submit(
        import
            .command_buffers
            .into_iter()
            .chain(std::iter::once(import.encoder.finish())),
    );
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("the DMA-BUF copy failed");
    drop(import.guard);
    frame.release(None);
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released]
    );
    assert_eq!(gpu.map(&readback), expected_texels(), "sampled texels");
}

#[test]
fn vulkan_copy_to_texture_imports_the_frame() {
    copy_to_texture_imports_the_frame_on(wgpu::Backends::VULKAN);
}

#[test]
fn vulkan_copy_into_orders_the_copy_before_caller_commands() {
    copy_into_orders_the_copy_before_caller_commands_on(wgpu::Backends::VULKAN, 3);
}

#[test]
fn gles_copy_to_texture_imports_the_frame() {
    copy_to_texture_imports_the_frame_on(wgpu::Backends::GL);
}

#[test]
fn gles_copy_into_orders_the_copy_before_caller_commands() {
    copy_into_orders_the_copy_before_caller_commands_on(wgpu::Backends::GL, 0);
}
