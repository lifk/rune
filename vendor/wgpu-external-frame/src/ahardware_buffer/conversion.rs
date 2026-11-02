//! The GPU pass that converts a 4:2:0 YCbCr buffer into a luma and a chroma
//! texture `wgpu` owns.
//!
//! A buffer reaches this pass when the driver describes it only through an
//! external format, which can be read only through a sampler carrying a
//! `VkSamplerYcbcrConversion`, or when the driver maps it to
//! `G8_B8R8_2PLANE_420_UNORM` and the device lacks
//! `wgpu::Features::TEXTURE_FORMAT_NV12` to alias it as `NV12`. `wgpu`
//! cannot express the sampler, so this pass is raw Vulkan: two render
//! passes that sample the buffer with the `RGB_IDENTITY` model — the stored
//! codes, unconverted — and write luma into an `R8Unorm` texture and the
//! Cb/Cr pairs into an `Rg8Unorm` texture at half the extent.
//!
//! The objects that depend only on the device ([`Shared`]) and those that
//! depend on the buffer's conversion parameters ([`ConversionPipeline`]) are
//! created once and cached; each frame creates only the views and
//! framebuffers that name its own images and the descriptor set that binds its
//! source, from a pool of its own ([`ConversionFrame`]).

use core::ops::Deref;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use ash::vk;
use wgpu::hal::Device as _;

use super::frame::FrameParts;
use super::vulkan::{self, BufferProperties, ImageFormat, ImageShape};
use super::{HardwareBufferImportError, HardwareBufferUsage, Retirement, Ycbcr420Planes};
use crate::YcbcrEncoding;

/// The luma plane's texture format, as `wgpu` and Vulkan name it.
pub(super) const LUMA_FORMAT: (wgpu::TextureFormat, vk::Format) =
    (wgpu::TextureFormat::R8Unorm, vk::Format::R8_UNORM);

/// The chroma plane's texture format, as `wgpu` and Vulkan name it.
pub(super) const CHROMA_FORMAT: (wgpu::TextureFormat, vk::Format) =
    (wgpu::TextureFormat::Rg8Unorm, vk::Format::R8G8_UNORM);

const VERTEX_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ycbcr_planes.vert.spv"));
const FRAGMENT_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ycbcr_planes.frag.spv"));

/// What a `VkSamplerYcbcrConversion` is built for, as the driver describes
/// the buffer.
///
/// A buffer with a Vulkan format is converted with a conversion built for
/// that format, which is how a defined-format `Y8Cb8Cr8_420` buffer imports
/// on a device without `wgpu::Features::TEXTURE_FORMAT_NV12`; a buffer the
/// driver maps to no Vulkan format is converted with one built for its
/// implementation-defined external format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionFormat {
    /// A Vulkan format, as a raw `VkFormat` value.
    Defined(i32),
    /// An implementation-defined external format.
    External(u64),
}

impl ConversionFormat {
    /// The format the driver's description of the buffer implies.
    const fn of(properties: &BufferProperties) -> Self {
        match properties.format {
            vk::Format::UNDEFINED => Self::External(properties.external_format),
            format => Self::Defined(format.as_raw()),
        }
    }

    /// The `VkFormat` the conversion and the source's view are created
    /// with: the buffer's own, or `UNDEFINED` for an external format.
    const fn vk_format(self) -> vk::Format {
        match self {
            Self::Defined(format) => vk::Format::from_raw(format),
            Self::External(_) => vk::Format::UNDEFINED,
        }
    }
}

impl core::fmt::Display for ConversionFormat {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Defined(format) => {
                write!(
                    formatter,
                    "Vulkan format {:?}",
                    vk::Format::from_raw(*format)
                )
            }
            Self::External(format) => write!(formatter, "external format {format:#x}"),
        }
    }
}

/// Imports the buffer of `parts` — of an external format, or of a defined
/// YCbCr format the device cannot alias — by converting it into a luma and
/// a chroma texture, and submits the conversion.
///
/// The conversion is submitted before this returns, so every later submission
/// that uses the planes runs after it. Once it has completed, the buffer and
/// every object the conversion created for this frame are released, and the
/// lease with them; the planes are ordinary `wgpu` textures from then on.
///
/// `hal_device` is dropped before the planes are handed to `wgpu`.
///
/// # Errors
///
/// Returns [`HardwareBufferImportError::ConversionDescriptorPool`] when the
/// descriptor set that binds the buffer cannot be allocated from a pool of
/// the converter's descriptor count. Nothing has been imported or created by
/// then, and `parts` — the lease with it — is dropped.
///
/// # Panics
///
/// Panics when the buffer is mipmapped or its extent is odd, which a 4:2:0
/// plane layout cannot represent, or when Vulkan fails to import or convert
/// it.
pub(super) fn import(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    converter: &mut Converter,
    hal_device: impl Deref<Target = wgpu::hal::vulkan::Device>,
    parts: FrameParts,
    properties: &BufferProperties,
    encoding: YcbcrEncoding,
) -> Result<Ycbcr420Planes, HardwareBufferImportError> {
    let FrameParts {
        buffer,
        description,
        acquire_fence,
        mut lease,
    } = parts;
    assert!(
        description.width.is_multiple_of(2)
            && description.height.is_multiple_of(2)
            && !description
                .usage
                .contains(HardwareBufferUsage::GPU_MIPMAP_COMPLETE),
        "a 4:2:0 YCbCr AHardwareBuffer must be a single mip level of even extent, but this one \
         is {}x{} with usage {:?}",
        description.width,
        description.height,
        description.usage,
    );
    let extent = vk::Extent2D {
        width: description.width,
        height: description.height,
    };
    let pipeline = converter.pipeline(properties);
    // The one step that can be refused, so it goes before anything that would
    // have to be unwound.
    let binding = SourceBinding::allocate(&pipeline)?;
    let imported = vulkan::import_buffer(
        &hal_device,
        buffer.buffer(),
        properties,
        &ImageShape {
            format: match pipeline.key.format {
                ConversionFormat::Defined(format) => ImageFormat::Defined {
                    format: vk::Format::from_raw(format),
                    // The conversion views the image whole, in its own
                    // format, so it needs neither MUTABLE_FORMAT nor
                    // EXTENDED_USAGE.
                    multi_planar: false,
                },
                ConversionFormat::External(external) => ImageFormat::External(external),
            },
            width: extent.width,
            height: extent.height,
            mip_levels: 1,
            usage: vk::ImageUsageFlags::SAMPLED,
        },
        acquire_fence,
    );
    let luma = PlaneTexture::new(&hal_device, LUMA_FORMAT.0, extent.width, extent.height);
    let chroma = PlaneTexture::new(
        &hal_device,
        CHROMA_FORMAT.0,
        extent.width / 2,
        extent.height / 2,
    );
    let frame = ConversionFrame::new(
        pipeline,
        binding,
        &ConversionTargets {
            source: imported.image(),
            luma: luma.image,
            chroma: chroma.image,
            extent,
        },
    );
    let source = imported.image();
    let acquire_semaphore = imported.acquire_semaphore();
    let raw_device = hal_device.raw_device().clone();
    let queue_family_index = hal_device.queue_family_index();
    drop(hal_device);
    let luma = luma.into_wgpu(device);
    let chroma = chroma.into_wgpu(device);
    if let Some(lease) = lease.as_mut() {
        lease.presented();
    }
    let conversion = super::record_raw(device, "wgpu_external_frame_ycbcr_conversion", |encoder| {
        // SAFETY: the encoder records into the submission below, which keeps
        // both planes alive until it completes, as `submit` describes; the
        // source image and `frame` are released only by the completion
        // callback registered below. The acquire barrier moves the source into
        // `SHADER_READ_ONLY_OPTIMAL` for the fragment shader before the
        // conversion reads it, and the planes were created just above, so no
        // submission has used them yet.
        unsafe {
            vulkan::record_acquire(&raw_device, encoder, source, queue_family_index);
            frame.record(encoder.raw_handle());
        }
    });
    let retirement = Retirement {
        imported,
        buffer,
        lease,
    };
    conversion.on_submitted_work_done(move || {
        // The views and framebuffers name the source image, so they go first.
        drop(frame);
        retirement.retire();
    });
    super::submit(
        device,
        queue,
        conversion,
        &[&luma, &chroma],
        acquire_semaphore,
    );
    Ok(Ycbcr420Planes {
        luma: luma.create_view(&wgpu::TextureViewDescriptor::default()),
        chroma: chroma.create_view(&wgpu::TextureViewDescriptor::default()),
        encoding,
    })
}

/// A plane texture created through `wgpu-hal`, before `wgpu` adopts it.
///
/// It is created through the hal layer rather than `wgpu::Device` because the
/// conversion writes it outside `wgpu`'s knowledge: `wgpu` adopts it as
/// already initialized, where a texture of its own would count as
/// uninitialized and be cleared before its first use, discarding the planes.
struct PlaneTexture {
    texture: wgpu::hal::vulkan::Texture,
    image: vk::Image,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
}

impl PlaneTexture {
    const LABEL: Option<&'static str> = Some("wgpu_external_frame_ycbcr_plane");
    /// Sampled and copied by the consumer, rendered to by the conversion.
    const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::RENDER_ATTACHMENT);

    fn new(
        hal_device: &wgpu::hal::vulkan::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let descriptor = wgpu::hal::TextureDescriptor {
            label: Self::LABEL,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUses::RESOURCE
                | wgpu::TextureUses::COPY_SRC
                | wgpu::TextureUses::COLOR_TARGET,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        };
        // SAFETY: the descriptor names a single-level 2D texture of a format
        // every device can sample, copy and render to, at a nonzero extent
        // within the device's limits since it is at most the buffer's.
        let texture = unsafe { hal_device.create_texture(&descriptor) }.unwrap_or_else(|error| {
            panic!("failed to create a {format:?} YCbCr plane texture: {error}")
        });
        // SAFETY: the texture was just created by `wgpu-hal` and owns its
        // image, which stays alive at least as long as the texture.
        let image = unsafe { texture.raw_handle() };
        Self {
            texture,
            image,
            format,
            size,
        }
    }

    /// Hands the texture to `wgpu`, which owns it from then on.
    fn into_wgpu(self, device: &wgpu::Device) -> wgpu::Texture {
        // SAFETY: the hal texture was created on this device's hal device from
        // a descriptor that matches this one, usage for usage. It counts as
        // initialized because the conversion, submitted before any use `wgpu`
        // records, writes every texel; its render passes leave it in
        // `SHADER_READ_ONLY_OPTIMAL`, made visible to the shader stages, which
        // is what `wgpu` assumes of the declared `RESOURCE` state.
        unsafe {
            device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                self.texture,
                &wgpu::TextureDescriptor {
                    label: Self::LABEL,
                    size: self.size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.format,
                    usage: Self::USAGE,
                    view_formats: &[],
                },
                wgpu::TextureUses::RESOURCE,
            )
        }
    }
}

/// The parameters a `VkSamplerYcbcrConversion` is created from, and so the
/// key its cached pipeline is found by.
///
/// The model is always `RGB_IDENTITY`, which ignores the range, and every
/// conversion is built with the identity component mapping — which an
/// external format ignores anyway — so neither is part of the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConversionKey {
    format: ConversionFormat,
    x_chroma_offset: vk::ChromaLocation,
    y_chroma_offset: vk::ChromaLocation,
}

/// Converts YCbCr buffers on one device, caching a pipeline per set of
/// conversion parameters.
pub(super) struct Converter {
    shared: Arc<Shared>,
    pipelines: HashMap<ConversionKey, Arc<ConversionPipeline>>,
}

impl core::fmt::Debug for Converter {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Converter")
            .field("pipelines", &self.pipelines.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Converter {
    /// Creates the device-wide objects of the conversion.
    ///
    /// `hal_device` must enable every one of [`super::DEVICE_EXTENSIONS`],
    /// which `HardwareBufferImporter::new` asserts, and the
    /// `samplerYcbcrConversion` feature, which every device opened with the
    /// [`super::DeviceRequirements`] has.
    pub(super) fn new(hal_device: &wgpu::hal::vulkan::Device) -> Self {
        Self {
            shared: Arc::new(Shared::new(
                hal_device.raw_device(),
                combined_image_sampler_descriptor_count(hal_device),
            )),
            pipelines: HashMap::new(),
        }
    }

    /// The pipeline that converts buffers with `properties`, created on first
    /// use.
    pub(super) fn pipeline(&mut self, properties: &BufferProperties) -> Arc<ConversionPipeline> {
        let key = ConversionKey {
            format: ConversionFormat::of(properties),
            x_chroma_offset: properties.x_chroma_offset,
            y_chroma_offset: properties.y_chroma_offset,
        };
        Arc::clone(
            self.pipelines.entry(key).or_insert_with(|| {
                Arc::new(ConversionPipeline::new(Arc::clone(&self.shared), key))
            }),
        )
    }
}

/// The conversion's objects that depend only on the device: the shaders, the
/// render passes that write each plane, and the size of the pool each frame
/// allocates its source's descriptor set from.
struct Shared {
    device: ash::Device,
    /// The pool descriptors one combined image sampler with an immutable
    /// YCbCr sampler may consume, from
    /// [`combined_image_sampler_descriptor_count`].
    descriptor_count: u32,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    luma_pass: vk::RenderPass,
    chroma_pass: vk::RenderPass,
}

impl Shared {
    fn new(device: &ash::Device, descriptor_count: u32) -> Self {
        let vertex = shader_module(device, VERTEX_SPIRV);
        let fragment = shader_module(device, FRAGMENT_SPIRV);
        Self {
            device: device.clone(),
            descriptor_count,
            vertex,
            fragment,
            luma_pass: plane_render_pass(device, LUMA_FORMAT.1),
            chroma_pass: plane_render_pass(device, CHROMA_FORMAT.1),
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: these objects were created on `self.device` by `new`, are
        // owned solely by this value and destroyed once. Every pipeline and
        // framebuffer created from them holds an `Arc` on this value, so none
        // is still alive, and so no pending submission uses them.
        unsafe {
            self.device.destroy_render_pass(self.chroma_pass, None);
            self.device.destroy_render_pass(self.luma_pass, None);
            self.device.destroy_shader_module(self.fragment, None);
            self.device.destroy_shader_module(self.vertex, None);
        }
    }
}

/// The number of pool descriptors to reserve for the conversion's one
/// combined image sampler, whose immutable sampler carries the YCbCr
/// conversion of an external format.
///
/// Such a descriptor may consume more than one pool descriptor — typically
/// one per plane. The specification gives the number for a `VkFormat` through
/// `VkSamplerYcbcrConversionImageFormatProperties::combinedImageSamplerDescriptorCount`,
/// which an external format cannot be queried for, and for every format the
/// implementation converts, external ones included, through
/// `maxCombinedImageSamplerDescriptorCount` ("sized to accommodate any and
/// all formats that require a sampler Y′CbCr conversion", the note under
/// `VkDescriptorPoolSize`). Only `VK_KHR_maintenance6` and Vulkan 1.4 report
/// that property, and on a `wgpu` device only the extension can: `wgpu-hal`
/// creates its instance with `VkApplicationInfo::apiVersion` at Vulkan 1.3
/// at most, and an application may use a physical device's core
/// functionality only up to that version. The property is
/// physical-device-level, so the physical device listing the extension is
/// enough; the device need not enable it.
///
/// Without the extension the count is 3, the most planes a 4:2:0 YCbCr
/// format has, which is the most any known implementation consumes for one
/// such descriptor. The specification does not guarantee that bound: a
/// driver that needs more makes the allocation fail, which the import reports
/// as [`HardwareBufferImportError::ConversionDescriptorPool`] rather than
/// retrying.
fn combined_image_sampler_descriptor_count(hal_device: &wgpu::hal::vulkan::Device) -> u32 {
    /// The most planes a 4:2:0 YCbCr format has.
    const PLANES_420: u32 = 3;
    let instance = hal_device.shared_instance().raw_instance();
    let physical_device = hal_device.raw_physical_device();
    // SAFETY: the physical device is the one the live device was created on,
    // from this instance.
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .unwrap_or_else(|error| {
            panic!("failed to enumerate the Vulkan device's extensions: {error}")
        });
    let has_maintenance6 = extensions
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(ash::khr::maintenance6::NAME));
    if !has_maintenance6 {
        return PLANES_420;
    }
    let mut maintenance6 = vk::PhysicalDeviceMaintenance6PropertiesKHR::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance6);
    // SAFETY: `vkGetPhysicalDeviceProperties2` is core in the Vulkan 1.1 the
    // device was required to have and the instance `wgpu-hal` creates for it.
    // The chained structure belongs to `VK_KHR_maintenance6`, which the
    // physical device was just seen to support, and both structures are
    // initialized locals the query writes through.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
    let count = maintenance6.max_combined_image_sampler_descriptor_count;
    // A pool size of zero descriptors is invalid, so a driver reporting it
    // must not reach `vkCreateDescriptorPool`.
    assert_ne!(
        count, 0,
        "the Vulkan driver reports a maxCombinedImageSamplerDescriptorCount of 0"
    );
    count
}

fn shader_module(device: &ash::Device, spirv: &[u8]) -> vk::ShaderModule {
    let code = ash::util::read_spv(&mut Cursor::new(spirv))
        .expect("the build script emits whole SPIR-V words");
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    // SAFETY: `code` is SPIR-V that `glslc` compiled for Vulkan 1.1 when the
    // crate was built, and `create_info` borrows it for the call.
    unsafe { device.create_shader_module(&create_info, None) }.unwrap_or_else(|error| {
        panic!("failed to create the YCbCr conversion shader module: {error}")
    })
}

/// A render pass with one color attachment of `format` that the pass
/// overwrites entirely and leaves ready for `wgpu` to sample.
fn plane_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        // Every texel is written, so the previous contents are not needed.
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        // The layout `wgpu` expects of a texture in the `RESOURCE` state,
        // which the plane textures are declared to start in.
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(core::slice::from_ref(&color));
    let dependencies = [
        // Orders the transition out of `UNDEFINED` and the writes after any
        // earlier use of the memory on this queue.
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        // Makes the writes visible to the stages and access `wgpu` assumes
        // for a texture in the `RESOURCE` state. A later `wgpu` barrier out
        // of that state names the same stages as its source, so it chains
        // after this dependency whatever the texture is used for next.
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(
                vk::PipelineStageFlags::VERTEX_SHADER
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
            )
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(core::slice::from_ref(&attachment))
        .subpasses(core::slice::from_ref(&subpass))
        .dependencies(&dependencies);
    // SAFETY: `create_info` and every array it points to are locals that
    // outlive the call. The one attachment is a color format every Vulkan
    // device can render to, referenced in `COLOR_ATTACHMENT_OPTIMAL`, and the
    // dependencies name only stages and accesses of the graphics pipeline.
    unsafe { device.create_render_pass(&create_info, None) }.unwrap_or_else(|error| {
        panic!("failed to create the YCbCr conversion render pass: {error}")
    })
}

/// The conversion's objects for one set of conversion parameters: the
/// `VkSamplerYcbcrConversion`, its immutable sampler, and the pipelines that
/// sample through it.
pub(super) struct ConversionPipeline {
    shared: Arc<Shared>,
    key: ConversionKey,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    luma: vk::Pipeline,
    chroma: vk::Pipeline,
}

impl ConversionPipeline {
    fn new(shared: Arc<Shared>, key: ConversionKey) -> Self {
        let device = &shared.device;
        // A zero external format is the structure's "no external format"
        // value, so chaining it for a defined-format conversion changes
        // nothing.
        let mut external_format =
            vk::ExternalFormatANDROID::default().external_format(match key.format {
                ConversionFormat::Defined(_) => 0,
                ConversionFormat::External(external) => external,
            });
        let conversion_info = vk::SamplerYcbcrConversionCreateInfo::default()
            .push_next(&mut external_format)
            .format(key.format.vk_format())
            // Passes the stored codes through: luma in green, Cb in blue, Cr
            // in red. The consumer applies the matrix the import reports.
            .ycbcr_model(vk::SamplerYcbcrModelConversion::RGB_IDENTITY)
            .ycbcr_range(vk::SamplerYcbcrRange::ITU_FULL)
            .components(vk::ComponentMapping::default())
            .x_chroma_offset(key.x_chroma_offset)
            .y_chroma_offset(key.y_chroma_offset)
            .chroma_filter(vk::Filter::NEAREST)
            .force_explicit_reconstruction(false);
        // SAFETY: the device was opened with the `samplerYcbcrConversion`
        // feature (`request_device`, `DeviceRequirements`). The struct and its
        // chained external format are locals that outlive the call. The format
        // is the one the driver reports for the buffer — a defined format,
        // `UNDEFINED` with its external format for a buffer it does not map —
        // and the chroma offsets are the ones the driver suggested for it, so
        // its format features support them.
        let conversion = unsafe { device.create_sampler_ycbcr_conversion(&conversion_info, None) }
            .unwrap_or_else(|error| {
                panic!(
                    "failed to create the VkSamplerYcbcrConversion for the {}: {error}",
                    key.format
                )
            });
        let mut conversion_binding =
            vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let sampler_info = vk::SamplerCreateInfo::default()
            .push_next(&mut conversion_binding)
            // A sampler with a YCbCr conversion must filter as the conversion's
            // chroma filter does, clamp to the edge, and use normalized
            // coordinates without anisotropy or comparison.
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .compare_enable(false)
            .unnormalized_coordinates(false);
        // SAFETY: the conversion was just created on `device`, and the sampler
        // parameters are the ones a sampler with a YCbCr conversion must have.
        let sampler =
            unsafe { device.create_sampler(&sampler_info, None) }.unwrap_or_else(|error| {
                panic!("failed to create the YCbCr conversion sampler: {error}")
            });
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .immutable_samplers(core::slice::from_ref(&sampler));
        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(core::slice::from_ref(&binding));
        // SAFETY: a YCbCr conversion sampler must be bound as an immutable
        // sampler of a combined image sampler, which this one binding is, and
        // `VUID-VkDescriptorSetLayoutBinding-descriptorType-12200` asks only
        // that every immutable sampler of the binding enable a conversion, as
        // this one does. The sampler was created on `device` just above.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
            .unwrap_or_else(|error| {
                panic!("failed to create the YCbCr conversion descriptor set layout: {error}")
            });
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(core::slice::from_ref(&set_layout));
        // SAFETY: the one set layout was created on `device` just above.
        let layout =
            unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap_or_else(|error| {
                panic!("failed to create the YCbCr conversion pipeline layout: {error}")
            });
        let [luma, chroma] = plane_pipelines(&shared, layout);
        Self {
            shared,
            key,
            conversion,
            sampler,
            set_layout,
            layout,
            luma,
            chroma,
        }
    }
}

impl Drop for ConversionPipeline {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: these objects were created on `device` by `new`, are owned
        // solely by this value and destroyed once, dependents first. Every
        // frame that records them holds an `Arc` on this value until the
        // submission it was recorded into has completed, so no pending
        // submission uses them.
        unsafe {
            device.destroy_pipeline(self.chroma, None);
            device.destroy_pipeline(self.luma, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler_ycbcr_conversion(self.conversion, None);
        }
    }
}

/// The luma and chroma pipelines: the same shaders, specialized by the plane
/// they write, each compatible with that plane's render pass.
fn plane_pipelines(shared: &Shared, layout: vk::PipelineLayout) -> [vk::Pipeline; 2] {
    let specialization_entry = vk::SpecializationMapEntry::default()
        .constant_id(0)
        .offset(0)
        .size(size_of::<vk::Bool32>());
    let luma_constant = vk::FALSE.to_ne_bytes();
    let chroma_constant = vk::TRUE.to_ne_bytes();
    let luma_specialization = vk::SpecializationInfo::default()
        .map_entries(core::slice::from_ref(&specialization_entry))
        .data(&luma_constant);
    let chroma_specialization = vk::SpecializationInfo::default()
        .map_entries(core::slice::from_ref(&specialization_entry))
        .data(&chroma_constant);
    let stages = |specialization| {
        [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shared.vertex)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shared.fragment)
                .name(c"main")
                .specialization_info(specialization),
        ]
    };
    let luma_stages = stages(&luma_specialization);
    let chroma_stages = stages(&chroma_specialization);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    // The viewport and scissor are dynamic, so one pipeline serves every
    // frame size.
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let luma_blend = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::R);
    let chroma_blend = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G);
    let luma_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(core::slice::from_ref(&luma_blend));
    let chroma_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(core::slice::from_ref(&chroma_blend));
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let common = vk::GraphicsPipelineCreateInfo::default()
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .dynamic_state(&dynamic)
        .layout(layout)
        .subpass(0);
    let create_infos = [
        common
            .stages(&luma_stages)
            .color_blend_state(&luma_blend_state)
            .render_pass(shared.luma_pass),
        common
            .stages(&chroma_stages)
            .color_blend_state(&chroma_blend_state)
            .render_pass(shared.chroma_pass),
    ];
    // SAFETY: every create info and the state it points to are locals that
    // outlive the call. The shader modules, render passes and layout were
    // created on `shared.device`; each pipeline targets subpass 0 of a render
    // pass with one color attachment, matching the fragment shader's one
    // output, and the layout's one set matches the shader's one binding.
    let pipelines = unsafe {
        shared
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
    }
    .unwrap_or_else(|(_, error)| {
        panic!("failed to create the YCbCr conversion pipelines: {error}")
    });
    pipelines
        .try_into()
        .expect("two create infos yield two pipelines")
}

/// The images one conversion reads and writes.
pub(super) struct ConversionTargets {
    /// The imported image, of the conversion's format.
    pub(super) source: vk::Image,
    /// The `R8Unorm` image, at the frame's extent.
    pub(super) luma: vk::Image,
    /// The `Rg8Unorm` image, at half the frame's extent.
    pub(super) chroma: vk::Image,
    /// The frame's extent; both dimensions are even.
    pub(super) extent: vk::Extent2D,
}

/// The descriptor set that binds one frame's source to the conversion's
/// immutable YCbCr sampler, and the pool it was allocated from.
///
/// Each frame allocates its set from a pool of its own, which is destroyed
/// with the frame once the submission that converts it has completed. No set
/// is shared between frames, so however many are in flight, none is updated
/// or freed while a pending command buffer reads it, and nothing needs to
/// track them. Recycling sets across frames would save a pool creation per
/// frame, but a set could only come back from the completion callback, the
/// pool would have to grow with the frames in flight, and an exhausted shared
/// pool could not tell a pool full of live sets from a descriptor count too
/// small for one set. A fresh pool sized for exactly one set can: when it
/// cannot hold that set, the count is the only possible cause.
struct SourceBinding {
    shared: Arc<Shared>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl SourceBinding {
    /// Allocates the set of `pipeline`'s layout from a new pool of the
    /// device's descriptor count.
    ///
    /// # Errors
    ///
    /// Returns [`HardwareBufferImportError::ConversionDescriptorPool`] when
    /// the pool cannot hold the set.
    fn allocate(pipeline: &ConversionPipeline) -> Result<Self, HardwareBufferImportError> {
        let shared = &pipeline.shared;
        let device = &shared.device;
        let size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: shared.descriptor_count,
        };
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(core::slice::from_ref(&size));
        // SAFETY: `pool_info` and the size it points to are locals that
        // outlive the call: room for one set, and a nonzero count of one
        // descriptor type, which `combined_image_sampler_descriptor_count`
        // guarantees.
        let pool =
            unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap_or_else(|error| {
                panic!("failed to create the YCbCr conversion descriptor pool: {error}")
            });
        let mut binding = Self {
            shared: Arc::clone(shared),
            pool,
            set: vk::DescriptorSet::null(),
        };
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(core::slice::from_ref(&pipeline.set_layout));
        // SAFETY: the pool was just created on `device` and has room for one
        // set; the layout was created on the same device and outlives the
        // call, as `pipeline` does.
        match unsafe { device.allocate_descriptor_sets(&allocate_info) } {
            Ok(sets) => {
                binding.set = sets[0];
                Ok(binding)
            }
            // A fresh pool sized for one set can be short of nothing but
            // descriptors: the binding consumes more than the count.
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                Err(HardwareBufferImportError::ConversionDescriptorPool {
                    format: pipeline.key.format,
                    descriptor_count: shared.descriptor_count,
                })
            }
            Err(error) => panic!(
                "failed to allocate the YCbCr conversion descriptor set for the {}: {error}",
                pipeline.key.format
            ),
        }
    }
}

impl Drop for SourceBinding {
    fn drop(&mut self) {
        // SAFETY: the pool was created on this device by `allocate`, is owned
        // solely by this value and destroyed once, which frees its set. A
        // binding is dropped either before anything recorded it, or with its
        // frame once the submission that recorded it has completed, so no
        // pending command buffer uses the set.
        unsafe { self.shared.device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// One frame's conversion objects: the views and framebuffers that name its
/// images, and the descriptor set that binds its source.
///
/// They are used only by the submission that records the conversion, and are
/// destroyed once it has completed.
pub(super) struct ConversionFrame {
    pipeline: Arc<ConversionPipeline>,
    binding: SourceBinding,
    extent: vk::Extent2D,
    source_view: vk::ImageView,
    luma_view: vk::ImageView,
    chroma_view: vk::ImageView,
    luma_framebuffer: vk::Framebuffer,
    chroma_framebuffer: vk::Framebuffer,
}

impl ConversionFrame {
    fn new(
        pipeline: Arc<ConversionPipeline>,
        binding: SourceBinding,
        targets: &ConversionTargets,
    ) -> Self {
        let device = &pipeline.shared.device;
        let mut conversion_binding =
            vk::SamplerYcbcrConversionInfo::default().conversion(pipeline.conversion);
        let source_view_info = vk::ImageViewCreateInfo::default()
            .push_next(&mut conversion_binding)
            .image(targets.source)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(pipeline.key.format.vk_format())
            .components(vk::ComponentMapping::default())
            .subresource_range(color_subresource());
        // SAFETY: `targets.source` is a live image created on this device by
        // `import_buffer` for this frame. A view of it must have the image's
        // own format — `UNDEFINED` for an external-format image — identity
        // swizzles, and the conversion created for the same format, which
        // `pipeline` was looked up by.
        let source_view = unsafe { device.create_image_view(&source_view_info, None) }
            .unwrap_or_else(|error| {
                panic!("failed to create the view of an external-format AHardwareBuffer: {error}")
            });
        // The sampler is ignored: the binding's immutable YCbCr sampler is
        // used instead.
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(source_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(binding.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(core::slice::from_ref(&image_info));
        // SAFETY: the set was just allocated with `pipeline`'s layout and no
        // command buffer has bound it. The write fills its one combined image
        // sampler, whose immutable sampler enables a YCbCr conversion, with
        // the view created with the same conversion, in the layout the acquire
        // barrier moves the image into before the conversion reads it.
        unsafe { device.update_descriptor_sets(core::slice::from_ref(&write), &[]) };
        let mut frame = Self {
            binding,
            extent: targets.extent,
            source_view,
            luma_view: vk::ImageView::null(),
            chroma_view: vk::ImageView::null(),
            luma_framebuffer: vk::Framebuffer::null(),
            chroma_framebuffer: vk::Framebuffer::null(),
            pipeline,
        };
        let chroma_extent = vk::Extent2D {
            width: targets.extent.width / 2,
            height: targets.extent.height / 2,
        };
        frame.luma_view = frame.plane_view(targets.luma, LUMA_FORMAT.1);
        frame.chroma_view = frame.plane_view(targets.chroma, CHROMA_FORMAT.1);
        frame.luma_framebuffer = frame.framebuffer(
            frame.pipeline.shared.luma_pass,
            frame.luma_view,
            targets.extent,
        );
        frame.chroma_framebuffer = frame.framebuffer(
            frame.pipeline.shared.chroma_pass,
            frame.chroma_view,
            chroma_extent,
        );
        frame
    }

    fn device(&self) -> &ash::Device {
        &self.pipeline.shared.device
    }

    fn plane_view(&self, image: vk::Image, format: vk::Format) -> vk::ImageView {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(color_subresource());
        // SAFETY: `image` is a live single-level, single-layer image of
        // `format`, created by `wgpu-hal` with color-attachment usage.
        unsafe { self.device().create_image_view(&create_info, None) }.unwrap_or_else(|error| {
            panic!("failed to create the view of a YCbCr plane texture: {error}")
        })
    }

    fn framebuffer(
        &self,
        render_pass: vk::RenderPass,
        view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> vk::Framebuffer {
        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(core::slice::from_ref(&view))
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        // SAFETY: `view` is a view of an image of exactly `extent`, in the
        // format of `render_pass`'s one attachment.
        unsafe { self.device().create_framebuffer(&create_info, None) }.unwrap_or_else(|error| {
            panic!("failed to create the framebuffer of a YCbCr plane: {error}")
        })
    }

    /// Records the conversion: luma, then chroma.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be recording, outside a render pass, into the
    /// submission that keeps this frame alive until it completes, after the
    /// barrier that acquires the source image into
    /// `SHADER_READ_ONLY_OPTIMAL` for the fragment shader. The plane images
    /// must not be in use by any pending submission.
    pub(super) unsafe fn record(&self, command_buffer: vk::CommandBuffer) {
        let chroma_extent = vk::Extent2D {
            width: self.extent.width / 2,
            height: self.extent.height / 2,
        };
        // SAFETY: the caller guarantees a recording command buffer outside a
        // render pass. The set was allocated with the pipeline layout's only
        // set layout and written by `new` with the source's view, and it stays
        // alive, unchanged, until this frame is dropped after the submission
        // completes. Both plane pipelines share this layout, so binding either
        // keeps the set bound for every draw below. The plane recordings rest
        // on the caller's contract, with each plane's framebuffer, pipeline
        // and extent belonging together.
        unsafe {
            self.device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                core::slice::from_ref(&self.binding.set),
                &[],
            );
            self.record_plane(
                command_buffer,
                self.luma_framebuffer,
                self.pipeline.shared.luma_pass,
                self.pipeline.luma,
                self.extent,
            );
            self.record_plane(
                command_buffer,
                self.chroma_framebuffer,
                self.pipeline.shared.chroma_pass,
                self.pipeline.chroma,
                chroma_extent,
            );
        }
    }

    /// # Safety
    ///
    /// As [`Self::record`], after the source's descriptor set has been bound,
    /// and `framebuffer`, `render_pass` and `pipeline` must be one plane's, at
    /// that plane's `extent`.
    unsafe fn record_plane(
        &self,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        render_pass: vk::RenderPass,
        pipeline: vk::Pipeline,
        extent: vk::Extent2D,
    ) {
        let device = self.device();
        let area = vk::Rect2D::default().extent(extent);
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(area);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a Vulkan image extent is far below the 2^24 an f32 holds exactly"
        )]
        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0);
        // SAFETY: the caller guarantees a recording command buffer outside a
        // render pass, and that the framebuffer, render pass and pipeline are
        // one plane's. The render area is the framebuffer's whole extent, the
        // load op needs no clear values, the dynamic viewport and scissor are
        // set before the draw, and the pipeline's one descriptor set has been
        // bound. The draw of three vertices needs no vertex buffer.
        unsafe {
            device.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[area]);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);
        }
    }
}

impl Drop for ConversionFrame {
    fn drop(&mut self) {
        let device = self.device();
        // SAFETY: every handle was created on this device by `new` — or is
        // still null if `new` panicked part-way, which the destroy calls
        // accept — is owned solely by this value and destroyed once,
        // dependents first. A frame is dropped only once the submission that
        // recorded it has completed, so the GPU no longer uses any of them.
        unsafe {
            device.destroy_framebuffer(self.chroma_framebuffer, None);
            device.destroy_framebuffer(self.luma_framebuffer, None);
            device.destroy_image_view(self.chroma_view, None);
            device.destroy_image_view(self.luma_view, None);
            device.destroy_image_view(self.source_view, None);
        }
    }
}

const fn color_subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
