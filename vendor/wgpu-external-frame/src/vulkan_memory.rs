//! Memory-type selection shared by the Vulkan external-memory imports.

use ash::vk;

/// Picks the memory type an imported allocation is placed in.
///
/// `type_bits` is the set of memory types both the image and the external
/// handle accept. A device-local type is preferred, since every import here is
/// read by the GPU; otherwise the lowest compatible index is used.
///
/// # Panics
///
/// Panics when `type_bits` names no memory type of the device.
pub fn select_memory_type(hal_device: &wgpu::hal::vulkan::Device, type_bits: u32) -> u32 {
    // SAFETY: the instance and physical device both come from the live hal
    // device guard the caller holds, so they are valid and belong together.
    // The query only reads them and returns a value.
    let properties = unsafe {
        hal_device
            .shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(hal_device.raw_physical_device())
    };
    let mut first = None;
    for index in 0..properties.memory_type_count {
        if type_bits & (1 << index) == 0 {
            continue;
        }
        first.get_or_insert(index);
        let memory_type = properties.memory_types
            [usize::try_from(index).expect("Vulkan memory index must fit usize")];
        if memory_type
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return index;
        }
    }
    first.unwrap_or_else(|| {
        panic!("no Vulkan memory type of this device is in the compatible set {type_bits:#x}")
    })
}
