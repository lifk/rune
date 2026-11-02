//! `AHardwareBuffer` import through
//! `VK_ANDROID_external_memory_android_hardware_buffer`.

use std::os::fd::{AsRawFd as _, IntoRawFd as _, OwnedFd};

use ash::vk;

/// Vulkan's view of a hardware buffer: an image bound to the buffer's memory.
pub(super) struct BufferProperties {
    pub(super) allocation_size: u64,
    pub(super) memory_type_bits: u32,
    /// The Vulkan format the buffer's format corresponds to, or `UNDEFINED`
    /// when it only has an implementation-defined external format.
    pub(super) format: vk::Format,
    pub(super) external_format: u64,
    /// The YCbCr model the driver suggests for sampling the buffer, which
    /// names the matrix its samples were encoded with.
    pub(super) suggested_model: vk::SamplerYcbcrModelConversion,
    /// The code range the driver suggests the samples span.
    pub(super) suggested_range: vk::SamplerYcbcrRange,
    /// Where the driver says the chroma samples sit horizontally.
    pub(super) x_chroma_offset: vk::ChromaLocation,
    /// Where the driver says the chroma samples sit vertically.
    pub(super) y_chroma_offset: vk::ChromaLocation,
}

/// The Vulkan objects one import created, destroyed together once the GPU has
/// finished with them.
pub(super) struct ImportedImage {
    device: ash::Device,
    image: vk::Image,
    memory: Option<vk::DeviceMemory>,
    acquire_semaphore: Option<vk::Semaphore>,
}

impl core::fmt::Debug for ImportedImage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ImportedImage")
            .field("image", &self.image)
            .field("memory", &self.memory)
            .field("acquire_semaphore", &self.acquire_semaphore)
            .finish_non_exhaustive()
    }
}

impl ImportedImage {
    pub(super) const fn image(&self) -> vk::Image {
        self.image
    }

    pub(super) const fn acquire_semaphore(&self) -> Option<vk::Semaphore> {
        self.acquire_semaphore
    }
}

impl Drop for ImportedImage {
    fn drop(&mut self) {
        // SAFETY: every handle here was created on `self.device` by this
        // module, is owned solely by this value, and is destroyed exactly once.
        // The image goes before the memory bound to it. None of them may still
        // be in use by the GPU: an `ImportedImage` is dropped either while the
        // import is still being built, before anything was submitted, or from
        // the drop callback `wgpu` runs when it destroys the adopted texture,
        // which it does only after every submission using the texture — the
        // import's own submission, which also waits on the semaphore, included
        // — has completed. Freeing the memory drops the reference Vulkan took
        // on the hardware buffer when it was imported.
        unsafe {
            self.device.destroy_image(self.image, None);
            if let Some(memory) = self.memory {
                self.device.free_memory(memory, None);
            }
            if let Some(semaphore) = self.acquire_semaphore {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
    }
}

/// Asserts that `hal_device` enabled every extension an import uses.
pub(super) fn validate_device(hal_device: &wgpu::hal::vulkan::Device) {
    for extension in super::DEVICE_EXTENSIONS {
        assert!(
            hal_device.enabled_device_extensions().contains(&extension),
            "the Vulkan device does not enable {extension:?}; open it with \
             `ahardware_buffer::request_device` or add `ahardware_buffer::DEVICE_EXTENSIONS`"
        );
    }
}

pub(super) fn buffer_properties(
    hal_device: &wgpu::hal::vulkan::Device,
    buffer: &ndk::hardware_buffer::HardwareBuffer,
) -> BufferProperties {
    let loader = ash::android::external_memory_android_hardware_buffer::Device::new(
        hal_device.shared_instance().raw_instance(),
        hal_device.raw_device(),
    );
    let mut format_properties = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
    let mut properties =
        vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut format_properties);
    // SAFETY: the extension was asserted enabled by `validate_device`, so the
    // loader's entry point exists. `buffer` is a live hardware buffer the frame
    // holds a reference on for the whole call, and `properties` with its
    // chained format struct are initialized locals the call writes through.
    unsafe {
        loader
            .get_android_hardware_buffer_properties(buffer.as_ptr().cast(), &mut properties)
            .unwrap_or_else(|error| {
                panic!("failed to query the AHardwareBuffer's Vulkan properties: {error}")
            });
    }
    BufferProperties {
        allocation_size: properties.allocation_size,
        memory_type_bits: properties.memory_type_bits,
        format: format_properties.format,
        external_format: format_properties.external_format,
        suggested_model: format_properties.suggested_ycbcr_model,
        suggested_range: format_properties.suggested_ycbcr_range,
        x_chroma_offset: format_properties.suggested_x_chroma_offset,
        y_chroma_offset: format_properties.suggested_y_chroma_offset,
    }
}

/// The format of the image created for an import.
pub(super) enum ImageFormat {
    /// A Vulkan format, which `wgpu` adopts the image as.
    Defined {
        format: vk::Format,
        multi_planar: bool,
    },
    /// The driver's implementation-defined external format, which only a
    /// sampler with a matching `VkSamplerYcbcrConversion` can read.
    External(u64),
}

/// What the image created for an import looks like.
pub(super) struct ImageShape {
    pub(super) format: ImageFormat,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_levels: u32,
    pub(super) usage: vk::ImageUsageFlags,
}

pub(super) fn import_buffer(
    hal_device: &wgpu::hal::vulkan::Device,
    buffer: &ndk::hardware_buffer::HardwareBuffer,
    properties: &BufferProperties,
    shape: &ImageShape,
    acquire_fence: Option<OwnedFd>,
) -> ImportedImage {
    let raw = hal_device.raw_device();
    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);
    let (format, flags, external_format) = match shape.format {
        // A plane view of a multi-planar image has a different format than
        // the image, which Vulkan only allows on a mutable-format image;
        // extended usage lets the plane views carry usages the multi-planar
        // format itself lacks. The hardware-buffer usage-equivalence table
        // maps neither flag to a buffer usage, so every buffer accepts them.
        ImageFormat::Defined {
            format,
            multi_planar: true,
        } => (
            format,
            vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE,
            0,
        ),
        ImageFormat::Defined {
            format,
            multi_planar: false,
        } => (format, vk::ImageCreateFlags::empty(), 0),
        // An external-format image has no Vulkan format and takes no creation
        // flags.
        ImageFormat::External(external_format) => (
            vk::Format::UNDEFINED,
            vk::ImageCreateFlags::empty(),
            external_format,
        ),
    };
    // A zero external format is the structure's "no external format" value,
    // so chaining it for a defined-format image changes nothing.
    let mut external_format = vk::ExternalFormatANDROID::default().external_format(external_format);
    let create_info = vk::ImageCreateInfo::default()
        .push_next(&mut external)
        .push_next(&mut external_format)
        .flags(flags)
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: shape.width,
            height: shape.height,
            depth: 1,
        })
        .mip_levels(shape.mip_levels)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(shape.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `create_info` is fully initialized and both `push_next` structs
    // are local `mut` bindings that outlive the call. The hardware-buffer
    // handle type and the external-format struct belong to an extension
    // `validate_device` asserted enabled. The parameters are the ones the
    // memory import's valid usage demands of a dedicated image: the buffer's
    // width, height and single layer, the Vulkan format the driver reported
    // for it — or `UNDEFINED` with its external format, no flags and sampled
    // usage only — optimal tiling, one mip level unless the buffer carries a
    // full chain, and usages that are either free or backed by the buffer's
    // `GPU_SAMPLED_IMAGE` usage, which the caller checked.
    let image = unsafe { raw.create_image(&create_info, None) }.unwrap_or_else(|error| {
        panic!("failed to create the Vulkan image for an AHardwareBuffer: {error}")
    });
    let mut imported = ImportedImage {
        device: raw.clone(),
        image,
        memory: None,
        acquire_semaphore: None,
    };
    let memory_type_index =
        crate::vulkan_memory::select_memory_type(hal_device, properties.memory_type_bits);
    let mut import =
        vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(buffer.as_ptr().cast());
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let allocation = vk::MemoryAllocateInfo::default()
        .push_next(&mut import)
        .push_next(&mut dedicated)
        .allocation_size(properties.allocation_size)
        .memory_type_index(memory_type_index);
    // SAFETY: `allocation` is fully initialized and both `push_next` structs
    // are local `mut` bindings outliving the call. The size is the one
    // `vkGetAndroidHardwareBufferPropertiesANDROID` reported for this buffer,
    // and the memory type was chosen from the bits it reported. The dedicated
    // image was created just above for exactly this buffer, as the hardware
    // buffer import requires. `buffer` is live for the call: the frame holds a
    // reference on it. On success Vulkan takes a reference of its own, released
    // when the memory is freed.
    let memory = unsafe { raw.allocate_memory(&allocation, None) }.unwrap_or_else(|error| {
        panic!("failed to import the AHardwareBuffer as Vulkan memory: {error}")
    });
    imported.memory = Some(memory);
    // SAFETY: `image` and `memory` were both just created on `raw` and neither
    // has been bound before. The memory is a dedicated allocation for this
    // image, so offset 0 is the only valid and required offset.
    unsafe { raw.bind_image_memory(image, memory, 0) }.unwrap_or_else(|error| {
        panic!("failed to bind the AHardwareBuffer's memory to its Vulkan image: {error}")
    });
    imported.acquire_semaphore = acquire_fence.map(|fence| import_acquire_fence(hal_device, fence));
    imported
}

/// Imports the producer's sync file into a binary semaphore whose next wait
/// consumes it.
fn import_acquire_fence(hal_device: &wgpu::hal::vulkan::Device, fence: OwnedFd) -> vk::Semaphore {
    let raw = hal_device.raw_device();
    // SAFETY: a default binary-semaphore create info with no extension structs
    // is always valid on a live device.
    let semaphore = unsafe { raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
        .unwrap_or_else(|error| {
            panic!("failed to create the semaphore for an AHardwareBuffer acquire fence: {error}")
        });
    let loader = ash::khr::external_semaphore_fd::Device::new(
        hal_device.shared_instance().raw_instance(),
        raw,
    );
    let import = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(fence.as_raw_fd());
    // SAFETY: `VK_KHR_external_semaphore_fd` was asserted enabled by
    // `validate_device`, so the loader's entry point exists. `semaphore` was
    // just created and has no pending operations. A sync-file import must be
    // temporary, which `TEMPORARY` makes it. The descriptor is open: it is
    // still owned by `fence`.
    match unsafe { loader.import_semaphore_fd(&import) } {
        Ok(()) => {
            // A successful import transfers the descriptor to Vulkan, so
            // ownership is given up here rather than closing it.
            let _transferred = fence.into_raw_fd();
            semaphore
        }
        Err(error) => {
            // SAFETY: the semaphore was created above, nothing else refers to
            // it, and it is destroyed exactly once. On failure the descriptor
            // was not taken, so `fence` still owns it and closes it on drop.
            unsafe { raw.destroy_semaphore(semaphore, None) };
            panic!("failed to import the AHardwareBuffer acquire fence: {error}")
        }
    }
}

/// Records the acquire of the imported image from the producer: an ownership
/// transfer from the foreign queue family that also moves the image into the
/// layout its readers expect — the one `wgpu` is told an aliasing texture
/// starts in, and the one the conversion samples.
///
/// # Safety
///
/// `encoder` must be the Vulkan encoder of the submission that performs the
/// import, and `image` must be the image of an [`ImportedImage`] that outlives
/// that submission.
pub(super) unsafe fn record_acquire(
    device: &ash::Device,
    encoder: &mut wgpu::hal::vulkan::CommandEncoder,
    image: vk::Image,
    queue_family_index: u32,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        // The image was bound to memory a producer already defined. Per the
        // specification's external resource sharing rules, the `UNDEFINED`
        // layout an image is created with is a placeholder for such memory,
        // and the acquire from the foreign queue family is what establishes
        // its real layout without discarding the contents.
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
        .dst_queue_family_index(queue_family_index)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(vk::REMAINING_MIP_LEVELS)
                .base_array_layer(0)
                .layer_count(1),
        );
    // SAFETY: the caller guarantees `encoder` records into the import's
    // submission and that `image` outlives it. The barrier names an image with
    // one layer, and `VK_EXT_queue_family_foreign` — asserted enabled by
    // `validate_device` — makes `QUEUE_FAMILY_FOREIGN_EXT` a valid source
    // family. The source stage is `ALL_COMMANDS`, so the barrier chains after
    // the acquire-fence semaphore wait at `ALL_COMMANDS`; the destination
    // stages and access are the ones `wgpu` uses for a texture in the
    // `RESOURCE` state, which is the state it is told an aliasing texture
    // starts in, and they include the fragment shader the conversion reads
    // the image from.
    unsafe {
        device.cmd_pipeline_barrier(
            encoder.raw_handle(),
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
