//! Android `AHardwareBuffer` import.
//!
//! Android producers — `AImageReader` with GPU usage, `MediaCodec`,
//! `SurfaceControl` — hand out frames as `AHardwareBuffer`s. Vulkan imports
//! one through `VK_ANDROID_external_memory_android_hardware_buffer`. When the
//! driver maps the buffer to a Vulkan format, `wgpu` adopts the resulting image
//! through its `hal` layer, so the returned texture *is* the producer's buffer:
//! no pixel is copied, by the CPU or the GPU — unless a YCbCr buffer cannot be
//! aliased, either because the driver describes it only through an external
//! format or because the device lacks `TEXTURE_FORMAT_NV12`, in which case a
//! GPU pass copies its planes into textures `wgpu` owns
//! ([Conversion](#conversion)); no pixel passes through the CPU either way.
//!
//! # Device
//!
//! The import needs device extensions `wgpu` does not enable by itself
//! ([`DEVICE_EXTENSIONS`]) and the `samplerYcbcrConversion` feature, with
//! which YCbCr buffers the device cannot alias are converted. Open the
//! device with [`request_device`], or add them with [`DeviceRequirements`]
//! in `wgpu-hal`'s device-creation callback when a renderer opens its device
//! itself.
//!
//! # Formats
//!
//! The format comes from the Vulkan driver's own reading of the buffer
//! (`vkGetAndroidHardwareBufferPropertiesANDROID`), not from the buffer's
//! description, and decides what [`HardwareBufferImporter::import`] returns:
//!
//! | Vulkan format the driver reports | [`ImportedHardwareBuffer`] | Textures |
//! | --- | --- | --- |
//! | `R8G8B8A8_UNORM` (`R8G8B8A8_UNORM`, `R8G8B8X8_UNORM` buffers) | [`Rgba`](ImportedHardwareBuffer::Rgba) | one `Rgba8Unorm` texture that aliases the buffer |
//! | `G8_B8R8_2PLANE_420_UNORM` (`Y8Cb8Cr8_420` buffers, when the driver maps them) | [`Ycbcr420`](ImportedHardwareBuffer::Ycbcr420) | the `Plane0` and `Plane1` views of one `NV12` texture that aliases the buffer, or an `R8Unorm` and an `Rg8Unorm` texture converted from it when the device lacks `TEXTURE_FORMAT_NV12` |
//! | `UNDEFINED`, with an implementation-defined *external format*, for a `Y8Cb8Cr8_420` buffer (camera frames and many allocated YCbCr buffers) | [`Ycbcr420`](ImportedHardwareBuffer::Ycbcr420) | an `R8Unorm` and an `Rg8Unorm` texture, converted from the buffer on the GPU |
//!
//! Both kinds of [`Ycbcr420Planes`] read the same way: a full-resolution luma
//! plane sampled as `R8Unorm`, a half-resolution plane of interleaved Cb/Cr
//! pairs sampled as `Rg8Unorm`, holding the stored codes, together with the
//! [`YcbcrEncoding`] the driver reports for the buffer.
//! The `NV12` alias needs `wgpu::Features::TEXTURE_FORMAT_NV12`, which is an
//! optimization, not a requirement: a device without it imports the same
//! buffer through the conversion below. The importer chooses once, when it is
//! created, from the device's features.
//!
//! # Conversion
//!
//! Many drivers describe 4:2:0 YCbCr buffers only through an external
//! format, which can be read only through a sampler carrying a
//! `VkSamplerYcbcrConversion`; on some drivers, such as the Mali-G715's,
//! that is every such buffer, camera frames included. `wgpu` cannot express
//! that sampler, so the import samples the buffer in a small raw Vulkan
//! pass, submitted on the importer's queue, that writes the stored codes
//! into two textures `wgpu` owns. The same pass serves a defined-format
//! `Y8Cb8Cr8_420` buffer on a device that cannot alias it as `NV12`. No
//! pixel passes through the CPU. The pass needs only the
//! `samplerYcbcrConversion` feature, which every import requires. It binds
//! the buffer through a descriptor set from a pool sized for one combined
//! image sampler with a YCbCr conversion:
//! `maxCombinedImageSamplerDescriptorCount` descriptors where the device
//! reports it (`VK_KHR_maintenance6`), and otherwise 3, one per plane, which
//! suffices on every known driver but is not guaranteed by the specification;
//! a driver that needs more is reported as
//! [`HardwareBufferImportError::ConversionDescriptorPool`]. Only
//! `Y8Cb8Cr8_420` buffers are converted, since
//! only their format says the samples are 8-bit 4:2:0; an external-format
//! buffer of any other format, such as
//! `YCbCr_P010`, is rejected with [`HardwareBufferImportError::ExternalFormat`].
//! The pass exists only because `wgpu` has no YCbCr sampler; once it gains
//! one, these buffers import directly.
//!
//! # Ownership
//!
//! A [`HardwareBufferFrame`] holds a reference on the buffer, the producer's
//! acquire fence, and its [`HardwareBufferLease`]. Importing consumes the
//! frame:
//!
//! 1. The lease is told the frame was
//!    [presented](HardwareBufferLease::presented), and the import submits the
//!    acquire of the buffer from the producer's (foreign) queue family, which
//!    waits on the acquire fence on the GPU — together with the conversion,
//!    for an external format.
//! 2. Once nothing reads the buffer any more, the Vulkan objects and the
//!    buffer reference are released, and finally the lease, which hands the
//!    buffer back to the producer. For an aliasing import that is when its
//!    texture has been dropped and every submission that used it has
//!    completed; for a converted one, when the conversion has completed,
//!    whatever the caller does with the planes.
//!
//! Neither path lets the GPU write the buffer, so the producer gets back
//! exactly what it wrote.

mod alias;
mod conversion;
mod device;
mod frame;
mod vulkan;

pub use conversion::ConversionFormat;
pub use device::{DEVICE_EXTENSIONS, DeviceRequestError, DeviceRequirements, request_device};
pub use frame::{HardwareBufferFrame, HardwareBufferLease};
pub use ndk::hardware_buffer::{HardwareBuffer, HardwareBufferDesc, HardwareBufferUsage};
pub use ndk::hardware_buffer_format::HardwareBufferFormat;

use ash::vk;
use conversion::Converter;
use frame::BufferReference;
use vulkan::{BufferProperties, ImportedImage};

use crate::{YcbcrEncoding, YcbcrMatrix, YcbcrRange};

/// Why a hardware buffer cannot be imported.
///
/// Every variant is a property of the buffer the producer handed over, or of
/// the device it was handed to, so the frame is dropped — releasing its lease
/// — when an import is rejected.
#[derive(Debug, thiserror::Error)]
pub enum HardwareBufferImportError {
    /// The driver describes the buffer only through an implementation-defined
    /// external format, and the buffer's own format is not `Y8Cb8Cr8_420`,
    /// the one whose samples the conversion knows to be 8-bit 4:2:0 YCbCr.
    /// Converting anything else into 8-bit planes could silently drop
    /// precision or misread the layout.
    #[error(
        "the AHardwareBuffer format {format:?} has no Vulkan format on this device, only the \
         external format {external_format:#x}, and only Y8Cb8Cr8_420 buffers of an external \
         format are converted"
    )]
    ExternalFormat {
        /// The buffer's own format.
        format: HardwareBufferFormat,
        /// The driver's implementation-defined format identifier.
        external_format: u64,
    },
    /// The buffer needs the YCbCr conversion, and the driver refused to
    /// allocate the descriptor set that binds it to its YCbCr sampler from a
    /// pool of `descriptor_count` combined image sampler descriptors.
    ///
    /// The count is the device's `maxCombinedImageSamplerDescriptorCount`
    /// when it reports one (`VK_KHR_maintenance6`), which the specification
    /// sizes for every format the device converts, and otherwise 3, the most
    /// planes a 4:2:0 format has, which the specification does not guarantee
    /// to suffice. On a device without the extension, this error means the
    /// driver consumes more than 3 descriptors for this format; on one with
    /// it, that the driver contradicts its own property. It is not retried.
    #[error(
        "the Vulkan driver could not allocate the descriptor set that binds the AHardwareBuffer's \
         {format} to its YCbCr sampler from a pool of {descriptor_count} combined image sampler \
         descriptors"
    )]
    ConversionDescriptorPool {
        /// What the conversion samples the buffer through: its Vulkan
        /// format, or its external format when the driver reports none.
        format: ConversionFormat,
        /// The pool's combined image sampler descriptors.
        descriptor_count: u32,
    },
    /// The driver suggests a YCbCr model for the buffer that is not one of
    /// the matrices a [`YcbcrEncoding`] names — such as `RGB_IDENTITY`, which
    /// means the buffer does not hold YCbCr at all.
    #[error(
        "the AHardwareBuffer format {format:?} imports as YCbCr, but the driver suggests the \
         YCbCr model {}, which names no YCbCr matrix",
        ycbcr_model_name(*.model)
    )]
    UnsupportedYcbcrModel {
        /// The buffer's own format.
        format: HardwareBufferFormat,
        /// The raw `VkSamplerYcbcrModelConversion` value the driver
        /// suggested.
        model: i32,
    },
    /// The buffer has a Vulkan format, but not one this import maps to a
    /// `wgpu` texture format.
    #[error(
        "the AHardwareBuffer format {format:?} imports as the Vulkan format {}, which this \
         import does not support",
        vk_format_name(*.vulkan_format)
    )]
    UnsupportedFormat {
        /// The buffer's own format.
        format: HardwareBufferFormat,
        /// The raw `VkFormat` value the driver reported.
        vulkan_format: i32,
    },
    /// The buffer was not allocated for GPU sampling.
    #[error("the AHardwareBuffer usage {0:?} lacks GPU_SAMPLED_IMAGE")]
    NotGpuSampled(HardwareBufferUsage),
    /// The buffer holds protected content, which only a protected device can
    /// read.
    #[error("the AHardwareBuffer holds protected content")]
    Protected,
    /// The buffer has more than one layer.
    #[error("the AHardwareBuffer has {0} layers; only single-layer buffers import as a 2D texture")]
    Layered(u32),
}

fn vk_format_name(raw: i32) -> String {
    format!("{:?}", vk::Format::from_raw(raw))
}

fn ycbcr_model_name(raw: i32) -> String {
    format!("{:?}", vk::SamplerYcbcrModelConversion::from_raw(raw))
}

/// A hardware buffer imported into textures on a `wgpu` device.
#[derive(Debug)]
pub enum ImportedHardwareBuffer {
    /// An `Rgba8Unorm` texture that aliases an RGBA buffer, usable as
    /// `TEXTURE_BINDING` and `COPY_SRC`. Check
    /// [`HardwareBufferFrame::force_opaque`] before importing to know whether
    /// its alpha is meaningful.
    Rgba(wgpu::Texture),
    /// The planes of a 4:2:0 YCbCr buffer.
    Ycbcr420(Ycbcr420Planes),
}

/// The planes of a 4:2:0 YCbCr buffer, and how to turn them into R'G'B'.
///
/// Both views sample the stored codes: no range expansion or matrix is
/// applied, which is left to the consumer, with [`Self::encoding`].
#[derive(Debug)]
pub struct Ycbcr420Planes {
    /// The full-resolution Y' plane, sampled as `R8Unorm`.
    pub luma: wgpu::TextureView,
    /// The half-resolution plane of interleaved Cb/Cr pairs, sampled as
    /// `Rg8Unorm` with Cb in red and Cr in green.
    pub chroma: wgpu::TextureView,
    /// The matrix and range the driver reports for the buffer.
    pub encoding: YcbcrEncoding,
}

/// Imports Android hardware buffers into textures on one `wgpu` device.
#[derive(Debug)]
pub struct HardwareBufferImporter {
    /// The YCbCr conversion, created by the first import that needs it.
    /// Declared first so it is dropped while the device is still held.
    converter: Option<Converter>,
    /// Whether the device has `wgpu::Features::TEXTURE_FORMAT_NV12`, which
    /// lets a defined-format `Y8Cb8Cr8_420` buffer alias as one `NV12`
    /// texture. A device's features are fixed at its creation, so which of
    /// aliasing and converting runs is decided once, here — never from a
    /// failed attempt.
    nv12: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl HardwareBufferImporter {
    /// Creates an importer for `device` and the queue imports are submitted
    /// on.
    ///
    /// The device must have been opened with the [`DeviceRequirements`]:
    /// through [`request_device`], or with [`DeviceRequirements::add_to`].
    ///
    /// # Panics
    ///
    /// Panics unless `device` is a Vulkan device with every one of
    /// [`DEVICE_EXTENSIONS`] enabled.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // SAFETY: `Device::as_hal` requires naming the device's real backend;
        // a non-Vulkan device yields `None` and panics here. The guard is only
        // read and dropped at the end of the statement.
        let hal_device = unsafe {
            device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("AHardwareBuffer import requires a Vulkan device")
        };
        vulkan::validate_device(&hal_device);
        drop(hal_device);
        Self {
            converter: None,
            nv12: device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_NV12),
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Imports `frame` into textures on the device, as the module-level table
    /// describes, and submits what the import records: the acquire of the
    /// buffer from its producer, and for an external format the conversion.
    ///
    /// Submissions made after this returns observe the producer's writes: the
    /// import submission waits on the frame's acquire fence on the GPU, and
    /// runs before them.
    ///
    /// # Errors
    ///
    /// Returns an error when the buffer is not GPU-sampled, protected,
    /// layered, of a format that has no `wgpu` equivalent, of an external
    /// format other than 8-bit 4:2:0 YCbCr, or of a YCbCr model that names no
    /// matrix, or when the buffer needs the YCbCr conversion and the driver
    /// cannot allocate its descriptor set
    /// ([`HardwareBufferImportError::ConversionDescriptorPool`]). The frame,
    /// and with it the lease, is dropped.
    ///
    /// # Panics
    ///
    /// Panics when a YCbCr buffer is odd-sized or mipmapped, which 4:2:0
    /// planes cannot represent, or when Vulkan fails to import it.
    pub fn import(
        &mut self,
        frame: HardwareBufferFrame,
    ) -> Result<ImportedHardwareBuffer, HardwareBufferImportError> {
        let parts = frame.into_parts();
        check_description(&parts.description)?;
        // SAFETY: `Device::as_hal` requires naming the device's real backend,
        // which `new` asserted is Vulkan, and that the raw device is not used
        // to invalidate `wgpu`'s state. It is used only to create and query
        // objects of this import's own.
        let hal_device = unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("AHardwareBuffer import requires a Vulkan device")
        };
        let properties = vulkan::buffer_properties(&hal_device, parts.buffer.buffer());
        match properties.format {
            vk::Format::UNDEFINED => {
                if parts.description.format != HardwareBufferFormat::Y8Cb8Cr8_420 {
                    return Err(HardwareBufferImportError::ExternalFormat {
                        format: parts.description.format,
                        external_format: properties.external_format,
                    });
                }
                let encoding = ycbcr_encoding(&parts.description, &properties)?;
                let converter = self
                    .converter
                    .get_or_insert_with(|| Converter::new(&hal_device));
                let planes = conversion::import(
                    &self.device,
                    &self.queue,
                    converter,
                    hal_device,
                    parts,
                    &properties,
                    encoding,
                )?;
                Ok(ImportedHardwareBuffer::Ycbcr420(planes))
            }
            vk::Format::R8G8B8A8_UNORM => {
                let texture = alias::import(
                    &self.device,
                    &self.queue,
                    hal_device,
                    parts,
                    &properties,
                    wgpu::TextureFormat::Rgba8Unorm,
                );
                Ok(ImportedHardwareBuffer::Rgba(texture))
            }
            vk::Format::G8_B8R8_2PLANE_420_UNORM => {
                let encoding = ycbcr_encoding(&parts.description, &properties)?;
                if self.nv12 {
                    let texture = alias::import(
                        &self.device,
                        &self.queue,
                        hal_device,
                        parts,
                        &properties,
                        wgpu::TextureFormat::NV12,
                    );
                    let plane = |format, aspect| {
                        texture.create_view(&wgpu::TextureViewDescriptor {
                            format: Some(format),
                            aspect,
                            ..wgpu::TextureViewDescriptor::default()
                        })
                    };
                    Ok(ImportedHardwareBuffer::Ycbcr420(Ycbcr420Planes {
                        luma: plane(wgpu::TextureFormat::R8Unorm, wgpu::TextureAspect::Plane0),
                        chroma: plane(wgpu::TextureFormat::Rg8Unorm, wgpu::TextureAspect::Plane1),
                        encoding,
                    }))
                } else {
                    let converter = self
                        .converter
                        .get_or_insert_with(|| Converter::new(&hal_device));
                    let planes = conversion::import(
                        &self.device,
                        &self.queue,
                        converter,
                        hal_device,
                        parts,
                        &properties,
                        encoding,
                    )?;
                    Ok(ImportedHardwareBuffer::Ycbcr420(planes))
                }
            }
            other => Err(HardwareBufferImportError::UnsupportedFormat {
                format: parts.description.format,
                vulkan_format: other.as_raw(),
            }),
        }
    }
}

/// Rejects buffers whose description rules out a 2D sampled import.
const fn check_description(
    description: &HardwareBufferDesc,
) -> Result<(), HardwareBufferImportError> {
    if !description
        .usage
        .contains(HardwareBufferUsage::GPU_SAMPLED_IMAGE)
    {
        return Err(HardwareBufferImportError::NotGpuSampled(description.usage));
    }
    if description
        .usage
        .contains(HardwareBufferUsage::PROTECTED_CONTENT)
    {
        return Err(HardwareBufferImportError::Protected);
    }
    if description.layers != 1 {
        return Err(HardwareBufferImportError::Layered(description.layers));
    }
    Ok(())
}

/// The encoding of a YCbCr buffer's samples, from the model and range the
/// driver suggests for sampling it.
fn ycbcr_encoding(
    description: &HardwareBufferDesc,
    properties: &BufferProperties,
) -> Result<YcbcrEncoding, HardwareBufferImportError> {
    let matrix = match properties.suggested_model {
        vk::SamplerYcbcrModelConversion::YCBCR_601 => YcbcrMatrix::Bt601,
        vk::SamplerYcbcrModelConversion::YCBCR_709 => YcbcrMatrix::Bt709,
        vk::SamplerYcbcrModelConversion::YCBCR_2020 => YcbcrMatrix::Bt2020,
        other => {
            return Err(HardwareBufferImportError::UnsupportedYcbcrModel {
                format: description.format,
                model: other.as_raw(),
            });
        }
    };
    let range = match properties.suggested_range {
        vk::SamplerYcbcrRange::ITU_FULL => YcbcrRange::Full,
        vk::SamplerYcbcrRange::ITU_NARROW => YcbcrRange::Video,
        other => panic!(
            "the Vulkan driver suggested {other:?} as the YCbCr range of an AHardwareBuffer, \
             which is not a VkSamplerYcbcrRange value"
        ),
    };
    Ok(YcbcrEncoding { matrix, range })
}

/// Records raw Vulkan commands into an encoder of their own.
///
/// `wgpu` forbids mixing raw and `wgpu` commands in one encoder, so the raw
/// commands of an import go into this one, and [`submit`] tracks the import's
/// textures in a second.
///
/// `record` must record only commands outside a render pass, or whole render
/// passes, and leave the encoder in a state `wgpu` can finish.
fn record_raw(
    device: &wgpu::Device,
    label: &'static str,
    record: impl FnOnce(&mut wgpu::hal::vulkan::CommandEncoder),
) -> wgpu::CommandEncoder {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    // SAFETY: recording raw Vulkan into a `wgpu` encoder requires naming its
    // real backend, which the importer asserted is Vulkan, and leaving the
    // encoder in a state `wgpu` can finish, which `record`'s contract
    // guarantees. No `wgpu` command touches this encoder.
    unsafe {
        encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
            record(encoder.expect("AHardwareBuffer command encoder is not Vulkan"));
        });
    }
    encoder
}

/// Submits `raw`, an import's raw commands, with a use of each of `textures`,
/// waiting first on the buffer's acquire semaphore.
fn submit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    raw: wgpu::CommandEncoder,
    textures: &[&wgpu::Texture],
    acquire_semaphore: Option<vk::Semaphore>,
) {
    let mut track = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_external_frame_ahardware_buffer_track"),
    });
    // A transition to the state the textures are already in records no
    // barrier, but it enters them in this submission's tracker, so `wgpu`
    // keeps them — and whatever their destruction releases — alive until the
    // submission completes, even if the caller drops them at once.
    track.transition_resources(
        core::iter::empty(),
        textures.iter().map(|&texture| wgpu::TextureTransition {
            texture,
            selector: None,
            state: wgpu::TextureUses::RESOURCE,
        }),
    );
    let command_buffers = [raw.finish(), track.finish()];
    if let Some(semaphore) = acquire_semaphore {
        // SAFETY: `Queue::as_hal` requires naming the queue's real backend,
        // which is the device's, Vulkan. The guard only stages a wait on the
        // semaphore for the queue's next submission, which is the `submit`
        // right below unless another thread submits first — in which case
        // that earlier submission waits instead, and this one still runs after
        // it, as `wgpu` orders submissions on its queue. The semaphore holds a
        // pending sync-file payload and is destroyed only after this
        // submission completes.
        let hal_queue = unsafe {
            queue
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("AHardwareBuffer import requires a Vulkan queue")
        };
        hal_queue.add_wait_semaphore(semaphore, None, vk::PipelineStageFlags::ALL_COMMANDS);
    }
    queue.submit(command_buffers);
}

/// Everything an import keeps alive for as long as the GPU reads its buffer.
struct Retirement {
    imported: ImportedImage,
    buffer: BufferReference,
    lease: Option<Box<dyn HardwareBufferLease>>,
}

impl Retirement {
    /// Releases the import in dependency order: Vulkan's objects (and the
    /// buffer reference Vulkan's memory holds), then this side's buffer
    /// reference, then the producer's lease.
    fn retire(self) {
        let Self {
            imported,
            buffer,
            lease,
        } = self;
        drop(imported);
        drop(buffer);
        if let Some(lease) = lease {
            lease.release();
        }
    }
}
