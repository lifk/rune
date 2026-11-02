//! Import of a buffer with a Vulkan format, as a texture that aliases it.

use core::ops::Deref;

use ash::vk;
use sync_wrapper::SyncWrapper;

use super::frame::FrameParts;
use super::vulkan::{self, BufferProperties, ImageFormat, ImageShape};
use super::{HardwareBufferUsage, Retirement};

/// Imports the buffer of `parts`, whose Vulkan format `wgpu` adopts as
/// `format`, as a texture that aliases it, and submits its acquire from the
/// producer.
///
/// `format` must need no feature the device lacks; the importer chooses it
/// from the device's features — `NV12` only when the device has
/// `wgpu::Features::TEXTURE_FORMAT_NV12` — so no check is needed here.
///
/// `hal_device` is dropped before the texture is handed to `wgpu`.
///
/// # Panics
///
/// Panics when the buffer's extent or mip chain is one `wgpu` cannot address
/// for `format`, or when Vulkan fails to import it.
pub(super) fn import(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    hal_device: impl Deref<Target = wgpu::hal::vulkan::Device>,
    parts: FrameParts,
    properties: &BufferProperties,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let FrameParts {
        buffer,
        description,
        acquire_fence,
        mut lease,
    } = parts;
    let layout = TextureLayout::new(&description, format);
    let imported = vulkan::import_buffer(
        &hal_device,
        buffer.buffer(),
        properties,
        &layout.image_shape(properties.format),
        acquire_fence,
    );
    let image = imported.image();
    let acquire_semaphore = imported.acquire_semaphore();
    let raw_device = hal_device.raw_device().clone();
    let queue_family_index = hal_device.queue_family_index();
    if let Some(lease) = lease.as_mut() {
        lease.presented();
    }
    let retirement = SyncWrapper::new(Retirement {
        imported,
        buffer,
        lease,
    });
    let drop_callback: wgpu::hal::DropCallback = Box::new(move || retirement.into_inner().retire());
    // SAFETY: `texture_from_raw` requires an image created to match the
    // descriptor, which it was: `image_shape` and `hal_descriptor` describe
    // the same extent, single layer, mip count and sample count, the image
    // has the Vulkan format `wgpu` maps the texture format to, and its
    // usages cover the descriptor's. `view_formats` is empty; the
    // multi-planar case is created mutable, as `wgpu` itself does for these
    // formats. With a drop callback the image stays this module's to
    // destroy, which the callback does through `Retirement`, and
    // `TextureMemory::External` leaves the memory to it as well.
    let hal_texture = unsafe {
        hal_device.texture_from_raw(
            image,
            &layout.hal_descriptor(),
            Some(drop_callback),
            wgpu::hal::vulkan::TextureMemory::External,
        )
    };
    drop(hal_device);
    // SAFETY: `hal_texture` was created on this device from a hal
    // descriptor that matches this one, and its memory holds the
    // producer's contents, so it counts as initialized. It is declared to
    // start in `RESOURCE`, which is the state the acquire barrier submitted
    // below leaves it in before any later submission can use it.
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            hal_texture,
            &layout.descriptor(),
            wgpu::TextureUses::RESOURCE,
        )
    };
    let acquire = super::record_raw(
        device,
        "wgpu_external_frame_ahardware_buffer_acquire",
        |encoder| {
            // SAFETY: the encoder records into the submission below, which keeps
            // `texture` — and through its drop callback `image` — alive until it
            // completes, as `submit` describes.
            unsafe { vulkan::record_acquire(&raw_device, encoder, image, queue_family_index) };
        },
    );
    super::submit(device, queue, acquire, &[&texture], acquire_semaphore);
    texture
}

/// The `wgpu` texture a buffer imports as.
struct TextureLayout {
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    mip_level_count: u32,
    usage: wgpu::TextureUsages,
}

impl TextureLayout {
    const LABEL: Option<&'static str> = Some("wgpu_external_frame_ahardware_buffer");

    fn new(description: &super::HardwareBufferDesc, format: wgpu::TextureFormat) -> Self {
        let size = wgpu::Extent3d {
            width: description.width,
            height: description.height,
            depth_or_array_layers: 1,
        };
        let mip_level_count = if description
            .usage
            .contains(HardwareBufferUsage::GPU_MIPMAP_COMPLETE)
        {
            size.max_mips(wgpu::TextureDimension::D2)
        } else {
            1
        };
        let multi_planar = format.is_multi_planar_format();
        let (width_multiple, height_multiple) = format.size_multiple_requirement();
        assert!(
            size.width.is_multiple_of(width_multiple)
                && size.height.is_multiple_of(height_multiple)
                && (!multi_planar || mip_level_count == 1),
            "a {format:?} texture must be a single mip level whose extent is a multiple of \
             {width_multiple}x{height_multiple}, but the AHardwareBuffer is {}x{} with \
             {mip_level_count} levels",
            size.width,
            size.height,
        );
        // `wgpu` only samples multi-planar textures; it cannot copy their
        // planes.
        let usage = if multi_planar {
            wgpu::TextureUsages::TEXTURE_BINDING
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC
        };
        Self {
            format,
            size,
            mip_level_count,
            usage,
        }
    }

    fn image_shape(&self, format: vk::Format) -> ImageShape {
        let mut usage = vk::ImageUsageFlags::SAMPLED;
        if self.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        ImageShape {
            format: ImageFormat::Defined {
                format,
                multi_planar: self.format.is_multi_planar_format(),
            },
            width: self.size.width,
            height: self.size.height,
            mip_levels: self.mip_level_count,
            usage,
        }
    }

    fn hal_descriptor(&self) -> wgpu::hal::TextureDescriptor<'static> {
        let mut usage = wgpu::TextureUses::RESOURCE;
        if self.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= wgpu::TextureUses::COPY_SRC;
        }
        wgpu::hal::TextureDescriptor {
            label: Self::LABEL,
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        }
    }

    const fn descriptor(&self) -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            label: Self::LABEL,
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: self.usage,
            view_formats: &[],
        }
    }
}
