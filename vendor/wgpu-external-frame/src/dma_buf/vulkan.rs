//! DMA-BUF import through Vulkan external memory.

use std::os::fd::{AsRawFd as _, IntoRawFd as _};

use ash::vk;
use sync_wrapper::SyncWrapper;

use super::frame::{DRM_FORMAT_MOD_INVALID, DmaBufFormat, DmaBufFrame, DmaBufPlane};

/// A `VkImage` aliasing an imported DMA-BUF, adopted by `wgpu` as an
/// initialized `COPY_SRC` texture and kept alive until the GPU work reading
/// it has completed.
pub(super) struct ImportedVulkanImage {
    /// The imported image adopted through `Device::create_texture_from_hal`.
    /// `wgpu` tracks it — and through the hal drop callback the `VkImage`
    /// and the memory it is bound to — until every submission referencing it
    /// has completed, which is what lets this value drop before or after a
    /// submission without ordering teardown itself.
    texture: wgpu::Texture,
    device: ash::Device,
    image: vk::Image,
    queue_family_index: u32,
}

impl core::fmt::Debug for ImportedVulkanImage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ImportedVulkanImage")
            .finish_non_exhaustive()
    }
}

// SAFETY: `ash::Device` is not `Send` only because it wraps the dispatchable
// `VkDevice` handle as a raw pointer; the image handle is a non-dispatchable
// `u64`. Vulkan permits a `VkDevice` to be used from any thread, and this
// type only ever reads `device` and `image` — their destruction happens in
// `Retirement`, which `wgpu`'s drop callback runs once the adopted texture
// is no longer referenced by any submission, on whatever thread performs
// that teardown. `Send` is what lets a deferred import be moved into wgpu's
// `on_submitted_work_done` callback, which is where a caller that defers the
// wait must drop it: that is the point at which the GPU has finished the
// copy that reads the image.
unsafe impl Send for ImportedVulkanImage {}

impl ImportedVulkanImage {
    /// The imported frame as a `COPY_SRC` texture for `copy_texture_to_texture`.
    ///
    /// `wgpu` counts textures created through `create_texture_from_hal` as
    /// initialized, so neither this texture nor a `wgpu` copy destination is
    /// lazily cleared around the copy — recording the copy through `wgpu`
    /// also marks the destination initialized for anything that uses it next.
    pub(super) const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Records the barrier that acquires `self.image` from
    /// `QUEUE_FAMILY_EXTERNAL` into this device's queue family, moving it to
    /// `TRANSFER_SRC_OPTIMAL` for the copy.
    ///
    /// # Safety contract
    ///
    /// `encoder` must be dedicated to raw commands — `wgpu` forbids mixing the
    /// two recording APIs in one — which callers uphold by handing over an
    /// encoder no `wgpu` command touches.
    pub(super) fn record_acquire(&self, encoder: &mut wgpu::CommandEncoder) {
        // SAFETY: recording raw Vulkan into a `wgpu` encoder. `as_hal_mut`
        // needs the right backend — checked, panicking on `None` — and the
        // commands must leave the encoder usable by `wgpu`: one pipeline
        // barrier is recorded, outside any render pass, allocating nothing.
        // `image` is owned by this import's hal texture, which outlives every
        // submission referencing it through `wgpu`'s tracking.
        unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
                let encoder = encoder.expect("DMA-BUF command encoder is not Vulkan");
                let acquire = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(self.image)
                    .subresource_range(subresource_range());
                self.device.cmd_pipeline_barrier(
                    encoder.raw_handle(),
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[acquire],
                );
            });
        }
    }

    /// Records the barrier releasing `self.image` back to
    /// `QUEUE_FAMILY_EXTERNAL` and `GENERAL`, the matched half of
    /// [`Self::record_acquire`], to run after the copy in the same
    /// submission.
    ///
    /// # Safety contract
    ///
    /// Same as [`Self::record_acquire`]: `encoder` must carry only raw
    /// commands.
    pub(super) fn record_release(&self, encoder: &mut wgpu::CommandEncoder) {
        // SAFETY: identical to `record_acquire` — one barrier, nothing else
        // recorded or allocated, so the encoder is exactly where `wgpu` left
        // it.
        unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
                let encoder = encoder.expect("DMA-BUF command encoder is not Vulkan");
                let release = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .image(self.image)
                    .subresource_range(subresource_range());
                self.device.cmd_pipeline_barrier(
                    encoder.raw_handle(),
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[release],
                );
            });
        }
    }
}

const fn subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// The Vulkan objects one import owns, released in dependency order by the
/// adopted texture's drop callback — once no submission references it any
/// more.
struct Retirement {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Retirement {
    fn retire(self) {
        // SAFETY: `image` and `memory` were created on `self.device` by
        // `import_dma_buf`, are owned solely by this value, and are destroyed
        // exactly once here — the image before the memory bound to it, as
        // Vulkan requires. Neither may still be in use by the GPU: the
        // callback runs when `wgpu` destroys the adopted texture, which it
        // does only after every submission referencing it has completed.
        // Freeing the memory also closes the DMA-BUF descriptor Vulkan took
        // ownership of at import.
        unsafe {
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

const fn hal_descriptor(frame: &DmaBufFrame) -> wgpu::hal::TextureDescriptor<'static> {
    wgpu::hal::TextureDescriptor {
        label: Some("wgpu_external_frame_dma_buf_import"),
        size: wgpu::Extent3d {
            width: frame.width,
            height: frame.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: frame.format.texture_format(),
        usage: wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    }
}

const fn wgpu_descriptor(frame: &DmaBufFrame) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("wgpu_external_frame_dma_buf_import"),
        size: wgpu::Extent3d {
            width: frame.width,
            height: frame.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: frame.format.texture_format(),
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

pub(super) fn import_dma_buf(
    device: &wgpu::Device,
    frame: &mut DmaBufFrame,
) -> ImportedVulkanImage {
    // SAFETY: `Device::as_hal` requires the named backend to be the device's
    // real one and that the exposed device is not used to invalidate wgpu's
    // state. The backend is checked: this function is reached only through the
    // importer's Vulkan path, which is selected for `wgpu::Backend::Vulkan`
    // alone, and a mismatch panics here instead of being reinterpreted. The raw
    // device is used only to create a new image and memory of this function's
    // own, never to touch anything wgpu owns.
    let hal_device = unsafe {
        device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("DMA-BUF Vulkan import requires a Vulkan device")
    };
    validate_import(&hal_device, frame.modifier);
    let raw = hal_device.raw_device().clone();
    let queue_family_index = hal_device.queue_family_index();
    let plane = frame
        .planes
        .pop()
        .expect("a packed DMA-BUF frame must contain one plane");
    let image = create_import_image(&raw, frame, &plane);
    let memory = import_image_memory(&hal_device, image, plane);
    // SAFETY: `image` and `memory` were both just created on `raw`, so they
    // belong to this device and neither has been bound before — this is the one
    // and only bind for each. `validate_import` established that the device
    // enables the external-memory and DRM-modifier extensions the pair was
    // created with. The memory was allocated from a type in
    // `vkGetImageMemoryRequirements(image).memoryTypeBits` (intersected with
    // what the descriptor supports) and sized to that requirement's `size`,
    // with a `VkMemoryDedicatedAllocateInfo` naming this exact image, so
    // offset 0 satisfies the alignment requirement by construction.
    if let Err(error) = unsafe { raw.bind_image_memory(image, memory, 0) } {
        // SAFETY: binding failed, so nothing owns these yet and neither is in
        // use by the GPU. Both are still live handles from `raw`, destroyed
        // exactly once here, memory after the image bound to it. Freeing the
        // memory closes the imported DMA-BUF descriptor Vulkan took over.
        unsafe {
            raw.free_memory(memory, None);
            raw.destroy_image(image, None);
        }
        panic!("failed to bind DMA-BUF Vulkan image memory: {error}");
    }
    let retirement = SyncWrapper::new(Retirement {
        device: raw.clone(),
        image,
        memory,
    });
    let drop_callback: wgpu::hal::DropCallback = Box::new(move || retirement.into_inner().retire());
    // SAFETY: `texture_from_raw` requires an image created to match the
    // descriptor, which it was: `create_import_image` and `hal_descriptor`
    // describe the same extent, mip count, sample count and dimension, the
    // image has the Vulkan format `wgpu` maps the texture format to, and its
    // `TRANSFER_SRC` usage covers `COPY_SRC`. `view_formats` is empty, as an
    // image created without `MUTABLE_FORMAT` demands. The drop callback keeps
    // the image — and `TextureMemory::External` the memory — this module's to
    // destroy, which `Retirement` does in dependency order.
    let hal_texture = unsafe {
        hal_device.texture_from_raw(
            image,
            &hal_descriptor(frame),
            Some(drop_callback),
            wgpu::hal::vulkan::TextureMemory::External,
        )
    };
    drop(hal_device);
    // SAFETY: `create_texture_from_hal` requires a hal texture created on this
    // device matching the descriptor, which `texture_from_raw` just produced.
    // Its memory holds the producer's contents, so it counts as initialized —
    // no `wgpu` pass will lazily clear it or the destination of a copy from
    // it. It is declared to start in `COPY_SRC`, the state the acquire barrier
    // submitted ahead of the copy leaves the image in, so `wgpu` records no
    // further transition for it.
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            hal_texture,
            &wgpu_descriptor(frame),
            wgpu::TextureUses::COPY_SRC,
        )
    };
    ImportedVulkanImage {
        texture,
        device: raw,
        image,
        queue_family_index,
    }
}

fn validate_import(hal_device: &wgpu::hal::vulkan::Device, modifier: u64) {
    assert!(
        hal_device
            .enabled_device_extensions()
            .contains(&ash::khr::external_memory_fd::NAME),
        "Vulkan device does not enable VK_KHR_external_memory_fd"
    );
    assert!(
        hal_device
            .enabled_device_extensions()
            .contains(&ash::ext::external_memory_dma_buf::NAME),
        "Vulkan device does not enable VK_EXT_external_memory_dma_buf"
    );
    assert!(
        hal_device
            .enabled_device_extensions()
            .contains(&ash::ext::image_drm_format_modifier::NAME),
        "Vulkan device does not enable VK_EXT_image_drm_format_modifier"
    );
    assert_ne!(
        modifier, DRM_FORMAT_MOD_INVALID,
        "Vulkan DMA-BUF import requires an explicit DRM format modifier"
    );
}

fn create_import_image(raw: &ash::Device, frame: &DmaBufFrame, plane: &DmaBufPlane) -> vk::Image {
    let handle_type = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
    let plane_layout = vk::SubresourceLayout {
        offset: u64::from(plane.offset),
        size: 0,
        row_pitch: u64::from(plane.stride),
        array_pitch: 0,
        depth_pitch: 0,
    };
    let mut modifier = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(frame.modifier)
        .plane_layouts(std::slice::from_ref(&plane_layout));
    let create_info = vk::ImageCreateInfo::default()
        .push_next(&mut external)
        .push_next(&mut modifier)
        .image_type(vk::ImageType::TYPE_2D)
        .format(match frame.format {
            DmaBufFormat::Bgra8 | DmaBufFormat::Bgrx8 => vk::Format::B8G8R8A8_UNORM,
            DmaBufFormat::Rgba8 | DmaBufFormat::Rgbx8 => vk::Format::R8G8B8A8_UNORM,
            DmaBufFormat::R8 => vk::Format::R8_UNORM,
            DmaBufFormat::R8G8 => vk::Format::R8G8_UNORM,
        })
        .extent(vk::Extent3D {
            width: frame.width,
            height: frame.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `create_info` is fully initialized above and its two `push_next`
    // extension structs — `VkExternalMemoryImageCreateInfo` and
    // `VkImageDrmFormatModifierExplicitCreateInfoEXT` — are local `mut`
    // bindings that outlive this call, as the borrow checker enforces through
    // `ImageCreateInfo`'s lifetime parameter. `plane_layout` likewise outlives
    // the borrow `plane_layouts` takes of it. The extensions those structs
    // require were asserted enabled by `validate_import`, which also rejected
    // `DRM_FORMAT_MOD_INVALID`, so `DRM_FORMAT_MODIFIER_EXT` tiling has the
    // explicit modifier it demands and the single plane layout matches the
    // single-plane packed format asserted at the frame's construction.
    unsafe { raw.create_image(&create_info, None) }
        .unwrap_or_else(|error| panic!("failed to create Vulkan DMA-BUF import image: {error}"))
}

fn import_image_memory(
    hal_device: &wgpu::hal::vulkan::Device,
    image: vk::Image,
    plane: DmaBufPlane,
) -> vk::DeviceMemory {
    let raw = hal_device.raw_device();
    let handle_type = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    // SAFETY: `image` was created on `raw` by the caller and has not been
    // destroyed, which is all `vkGetImageMemoryRequirements` requires; it only
    // reads the image and writes the returned struct.
    let requirements = unsafe { raw.get_image_memory_requirements(image) };
    let loader =
        ash::khr::external_memory_fd::Device::new(hal_device.shared_instance().raw_instance(), raw);
    let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
    // SAFETY: `VK_KHR_external_memory_fd` was asserted enabled by
    // `validate_import`, so the loader's entry point exists. The descriptor is
    // borrowed from `plane`, which still owns it here, so it is open for the
    // call; the query does not consume it. `fd_properties` is an initialized
    // local the call writes through. On failure the image created by the caller
    // is destroyed before panicking — it is not yet bound to any memory and
    // nothing else refers to it.
    unsafe {
        loader
            .get_memory_fd_properties(handle_type, plane.fd.as_raw_fd(), &mut fd_properties)
            .unwrap_or_else(|error| {
                raw.destroy_image(image, None);
                panic!("failed to query DMA-BUF Vulkan memory properties: {error}")
            });
    }
    let type_bits = requirements.memory_type_bits & fd_properties.memory_type_bits;
    assert!(
        type_bits != 0,
        "the DMA-BUF is incompatible with every Vulkan memory type"
    );
    let memory_type_index = crate::vulkan_memory::select_memory_type(hal_device, type_bits);
    let imported_fd = plane.fd.into_raw_fd();
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(handle_type)
        .fd(imported_fd);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let allocation = vk::MemoryAllocateInfo::default()
        .push_next(&mut import)
        .push_next(&mut dedicated)
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    // SAFETY: `allocation` is fully initialized and its two `push_next` structs
    // are local `mut` bindings outliving the call. `VK_KHR_external_memory_fd`
    // and `VK_EXT_external_memory_dma_buf` were asserted enabled by
    // `validate_import`, so `DMA_BUF_EXT` is an accepted handle type.
    // `memory_type_index` was chosen from `type_bits`, the intersection of the
    // image's requirements with the types the descriptor supports, which was
    // asserted non-empty. `imported_fd` is open and, per the Vulkan spec, is
    // transferred to the implementation on success — which is why nothing
    // closes it afterwards and why `plane.fd` was consumed with `into_raw_fd`
    // rather than borrowed.
    match unsafe { raw.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: on failure the implementation did *not* take the
            // descriptor, so this side still owns it and must close it exactly
            // once; `imported_fd` has not been closed and no `OwnedFd` holds it
            // any more, `into_raw_fd` having released it. The image is a live
            // handle from `raw`, unbound and unused by the GPU, destroyed once.
            unsafe {
                libc::close(imported_fd);
                raw.destroy_image(image, None);
            }
            panic!("failed to import the DMA-BUF as Vulkan memory: {error}");
        }
    }
}
