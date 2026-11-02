//! On-device tests of `AHardwareBuffer` import.
//!
//! Each test allocates a hardware buffer, writes a known pattern into it from
//! the CPU, imports it, and reads it back through a compute shader that samples
//! the imported texture or planes, so what is compared is what the GPU sees.
//! One test hands the import a sync file a GPU submission signals, standing in
//! for a producer's acquire fence.
#![cfg(target_os = "android")]

use std::os::fd::{FromRawFd as _, OwnedFd};
use std::sync::mpsc;

use ash::vk;

use ndk::hardware_buffer::HardwareBufferRef;

use wgpu_external_frame::ahardware_buffer::{
    DeviceRequirements, HardwareBuffer, HardwareBufferDesc, HardwareBufferFormat,
    HardwareBufferFrame, HardwareBufferImportError, HardwareBufferImporter, HardwareBufferLease,
    HardwareBufferUsage, ImportedHardwareBuffer, Ycbcr420Planes, request_device,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    importer: HardwareBufferImporter,
    pipeline: wgpu::ComputePipeline,
    spin_pipeline: wgpu::ComputePipeline,
}

fn vulkan_adapter() -> wgpu::Adapter {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("the device has no Vulkan adapter")
}

/// Requests `TEXTURE_FORMAT_NV12` only where the adapter offers it, so the
/// tests run on devices without it, like the emulator's `MoltenVK`.
fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        label: Some("ahardware_buffer_test"),
        required_features: adapter
            .features()
            .intersection(wgpu::Features::TEXTURE_FORMAT_NV12),
        ..wgpu::DeviceDescriptor::default()
    }
}

impl Gpu {
    /// A device opened by `request_device`.
    fn new() -> Self {
        let adapter = vulkan_adapter();
        let (device, queue) = request_device(&adapter, &device_descriptor(&adapter))
            .expect("failed to open a device that imports hardware buffers");
        Self::on(device, queue)
    }

    /// A device opened by `request_device` without `TEXTURE_FORMAT_NV12`,
    /// even on an adapter that offers it, so no buffer can alias as `NV12`.
    fn without_nv12() -> Self {
        let adapter = vulkan_adapter();
        let (device, queue) = request_device(
            &adapter,
            &wgpu::DeviceDescriptor {
                label: Some("ahardware_buffer_test"),
                required_features: wgpu::Features::empty(),
                ..wgpu::DeviceDescriptor::default()
            },
        )
        .expect("failed to open a device that imports hardware buffers");
        Self::on(device, queue)
    }

    /// A device opened through `wgpu-hal`'s creation callback with only the
    /// [`DeviceRequirements`], as a renderer that opens its device itself
    /// does, and checked to lack `VK_KHR_push_descriptor` and
    /// `VK_KHR_maintenance6`, neither of which the conversion needs.
    fn with_requirements_only() -> Self {
        let adapter = vulkan_adapter();
        let descriptor = device_descriptor(&adapter);
        // SAFETY: the adapter is a Vulkan adapter, and the guard is only used
        // to open a device.
        let hal_adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
            .expect("the adapter is a Vulkan adapter");
        let mut requirements =
            DeviceRequirements::new(&hal_adapter).expect("the adapter imports hardware buffers");
        // SAFETY: the features and limits are the defaults plus
        // `TEXTURE_FORMAT_NV12` where the adapter offers it, which
        // `device_descriptor` checked. The callback only adds what
        // `DeviceRequirements::new` checked the adapter supports, and
        // `requirements` outlives the call.
        let open_device = unsafe {
            hal_adapter.open_with_callback(
                descriptor.required_features,
                &descriptor.required_limits,
                &descriptor.memory_hints,
                Some(Box::new(|mut arguments| {
                    requirements.add_to(&mut arguments);
                })),
            )
        }
        .expect("failed to open a device with the import's requirements");
        for extension in [
            ash::khr::push_descriptor::NAME,
            ash::khr::maintenance6::NAME,
        ] {
            assert!(
                !open_device
                    .device
                    .enabled_device_extensions()
                    .contains(&extension),
                "the device was opened with {extension:?}, so it does not show the conversion \
                 works without it"
            );
        }
        drop(hal_adapter);
        // SAFETY: the device was opened just above from this adapter with
        // exactly `descriptor`'s features, limits and memory hints.
        let (device, queue) = unsafe { adapter.create_device_from_hal(open_device, &descriptor) }
            .expect("failed to adopt the device");
        Self::on(device, queue)
    }

    fn on(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let importer = HardwareBufferImporter::new(&device, &queue);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ahardware_buffer_read_texels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ahardware_buffer.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ahardware_buffer_read_texels"),
            layout: None,
            module: &module,
            entry_point: Some("read_texels"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let spin_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ahardware_buffer_spin"),
            source: wgpu::ShaderSource::Wgsl(include_str!("spin.wgsl").into()),
        });
        let spin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ahardware_buffer_spin"),
            layout: None,
            module: &spin_module,
            entry_point: Some("spin"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            importer,
            pipeline,
            spin_pipeline,
        }
    }

    /// Submits a compute dispatch that keeps the GPU busy for a while.
    fn spin(&self) {
        const INVOCATIONS: u32 = 64 * 64;
        let sink = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ahardware_buffer_spin"),
            size: u64::from(INVOCATIONS) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ahardware_buffer_spin"),
            layout: &self.spin_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sink.as_entire_binding(),
            }],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.spin_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(INVOCATIONS / 64, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Samples every texel of `view` (`width` × `height`) on the GPU and
    /// returns them as packed RGBA8 values, red in the low byte.
    fn read_texels(&self, view: &wgpu::TextureView, width: u32, height: u32) -> Vec<u32> {
        let size = u64::from(width * height) * 4;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ahardware_buffer_texels"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ahardware_buffer_texels"),
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        let readback = self.readback_buffer(size);
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        self.map(&readback)
    }

    /// Copies every texel of an RGBA8 `texture` into a buffer and returns them
    /// packed as [`Self::read_texels`] does.
    fn copy_texels(&self, texture: &wgpu::Texture) -> Vec<u32> {
        let size = texture.size();
        let bytes_per_row = size.width * 4;
        assert_eq!(
            bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
            0,
            "the test extent must copy without row padding"
        );
        let readback = self.readback_buffer(u64::from(bytes_per_row * size.height));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        self.queue.submit([encoder.finish()]);
        self.map(&readback)
    }

    fn readback_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ahardware_buffer_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn map(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let slice = buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("the map result receiver is alive");
        });
        self.wait();
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

    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("waiting for the GPU failed");
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LeaseEvent {
    Presented,
    Released,
}

#[derive(Debug)]
struct RecordingLease(mpsc::Sender<LeaseEvent>);

impl HardwareBufferLease for RecordingLease {
    fn presented(&mut self) {
        self.0
            .send(LeaseEvent::Presented)
            .expect("the lease event receiver is alive");
    }

    fn release(self: Box<Self>) {
        self.0
            .send(LeaseEvent::Released)
            .expect("the lease event receiver is alive");
    }
}

fn allocate(format: HardwareBufferFormat) -> HardwareBufferRef {
    HardwareBuffer::allocate(HardwareBufferDesc {
        width: WIDTH,
        height: HEIGHT,
        layers: 1,
        format,
        usage: HardwareBufferUsage::GPU_SAMPLED_IMAGE | HardwareBufferUsage::CPU_WRITE_OFTEN,
        stride: 0,
    })
    .unwrap_or_else(|error| panic!("failed to allocate a {format:?} hardware buffer: {error}"))
}

fn frame(
    buffer: &HardwareBuffer,
    fence: Option<OwnedFd>,
    lease: RecordingLease,
) -> HardwareBufferFrame {
    HardwareBufferFrame::new(buffer, fence).with_lease(Box::new(lease))
}

fn rgba_texel(x: u32, y: u32) -> [u8; 4] {
    [
        u8::try_from(x * 4).expect("fits"),
        u8::try_from(y * 5).expect("fits"),
        u8::try_from((x + y) * 2).expect("fits"),
        u8::try_from(255 - x).expect("fits"),
    ]
}

fn luma(x: u32, y: u32) -> u8 {
    u8::try_from((x * 3 + y * 2) % 256).expect("fits")
}

fn chroma(x: u32, y: u32) -> (u8, u8) {
    (
        u8::try_from(16 + x * 4 + y).expect("fits"),
        u8::try_from(240 - x * 3 - y * 2).expect("fits"),
    )
}

const fn pack(texel: [u8; 4]) -> u32 {
    u32::from_le_bytes(texel)
}

/// Writes [`rgba_texel`] into every pixel of an `R8G8B8A8_UNORM` buffer and
/// unlocks it synchronously.
fn write_rgba(buffer: &HardwareBuffer) {
    let stride = buffer.describe().stride;
    let pixels = buffer
        .lock(HardwareBufferUsage::CPU_WRITE_OFTEN, None, None)
        .expect("failed to lock the RGBA buffer")
        .cast::<u8>();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = usize::try_from((y * stride + x) * 4).expect("fits usize");
            // SAFETY: the lock maps `stride` × `HEIGHT` four-byte pixels, and
            // `x < WIDTH <= stride`, `y < HEIGHT`, so the four bytes written
            // lie inside the mapping, which stays valid until the unlock.
            unsafe {
                pixels
                    .add(offset)
                    .copy_from_nonoverlapping(rgba_texel(x, y).as_ptr(), 4);
            }
        }
    }
    buffer.unlock().expect("failed to unlock the RGBA buffer");
}

fn expected_rgba() -> Vec<u32> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| pack(rgba_texel(x, y))))
        .collect()
}

/// Writes [`luma`] and [`chroma`] into every sample of a `Y8Cb8Cr8_420`
/// buffer through its CPU-locked planes and unlocks it synchronously.
fn write_ycbcr(buffer: &HardwareBuffer) {
    let planes: Vec<_> = buffer
        .lock_planes(HardwareBufferUsage::CPU_WRITE_OFTEN, None, None)
        .expect("failed to lock the YCbCr buffer's planes")
        .collect();
    assert_eq!(
        planes.len(),
        3,
        "a YCbCr buffer locks as Y, Cb and Cr planes"
    );
    let write = |plane: usize, x: u32, y: u32, value: u8| {
        let plane = planes[plane];
        let offset = usize::try_from(y * plane.bytes_per_stride + x * plane.bytes_per_pixel)
            .expect("fits usize");
        // SAFETY: `lock_planes` maps each plane with the row and pixel strides
        // it reports, and every caller stays within that plane's extent — the
        // full extent for Y, half of it in each direction for Cb and Cr — so
        // the byte written lies inside the mapping, valid until the unlock.
        unsafe { plane.virtual_address.cast::<u8>().add(offset).write(value) };
    };
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            write(0, x, y, luma(x, y));
        }
    }
    for y in 0..HEIGHT / 2 {
        for x in 0..WIDTH / 2 {
            let (cb, cr) = chroma(x, y);
            write(1, x, y, cb);
            write(2, x, y, cr);
        }
    }
    buffer.unlock().expect("failed to unlock the YCbCr buffer");
}

fn expected_luma() -> Vec<u32> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| pack([luma(x, y), 0, 0, 255])))
        .collect()
}

fn expected_chroma() -> Vec<u32> {
    (0..HEIGHT / 2)
        .flat_map(|y| {
            (0..WIDTH / 2).map(move |x| {
                let (cb, cr) = chroma(x, y);
                pack([cb, cr, 0, 255])
            })
        })
        .collect()
}

/// The Vulkan format the driver reports for `buffer`, or `UNDEFINED` when it
/// describes the buffer only through an external format — what decides which
/// import path the buffer takes.
fn driver_format(gpu: &Gpu, buffer: &HardwareBuffer) -> vk::Format {
    // SAFETY: the test device is a Vulkan device; the guard is only used to
    // query properties of the test's own buffer.
    let hal_device = unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() }
        .expect("the test device is Vulkan");
    let loader = ash::android::external_memory_android_hardware_buffer::Device::new(
        hal_device.shared_instance().raw_instance(),
        hal_device.raw_device(),
    );
    let mut format_properties = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
    let mut properties =
        vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut format_properties);
    // SAFETY: `buffer` is live for the call and `properties` with its chained
    // format struct are initialized locals the call writes through; the
    // extension is enabled on the device.
    unsafe {
        loader
            .get_android_hardware_buffer_properties(buffer.as_ptr().cast(), &mut properties)
            .expect("failed to query the buffer's Vulkan properties");
    }
    format_properties.format
}

fn read_planes(gpu: &Gpu, planes: &Ycbcr420Planes) {
    assert_eq!(planes.luma.texture().width(), WIDTH);
    assert_eq!(planes.luma.texture().height(), HEIGHT);
    // The conversion samples with the identity model and nearest filtering,
    // so every code must survive exactly: the tolerance is zero.
    assert_eq!(
        gpu.read_texels(&planes.luma, WIDTH, HEIGHT),
        expected_luma(),
        "luma plane"
    );
    assert_eq!(
        gpu.read_texels(&planes.chroma, WIDTH / 2, HEIGHT / 2),
        expected_chroma(),
        "chroma plane"
    );
}

#[test]
fn rgba_buffer_imports_with_its_texels() {
    assert_rgba_imports(&mut Gpu::new());
}

/// A device opened with nothing but the import's requirements — no
/// `VK_KHR_push_descriptor` — imports RGBA buffers, and converts
/// `Y8Cb8Cr8_420` buffers of an external format with several conversions in
/// flight at once, each binding its own buffer.
#[test]
fn device_without_push_descriptors_converts_frames_in_flight() {
    let mut gpu = Gpu::with_requirements_only();
    assert_rgba_imports(&mut gpu);

    let buffers = [
        allocate(HardwareBufferFormat::Y8Cb8Cr8_420),
        allocate(HardwareBufferFormat::Y8Cb8Cr8_420),
    ];
    // Spinning first keeps the GPU busy, so the first conversion is still
    // pending when the second is submitted.
    gpu.spin();
    let (imports, receivers): (Vec<_>, Vec<_>) = buffers
        .iter()
        .map(|buffer| {
            write_ycbcr(buffer);
            let (events, received) = mpsc::channel();
            let ImportedHardwareBuffer::Ycbcr420(planes) = gpu
                .importer
                .import(frame(buffer, None, RecordingLease(events)))
                .expect("a Y8Cb8Cr8_420 buffer imports without push descriptors")
            else {
                panic!("a Y8Cb8Cr8_420 buffer imports as YCbCr planes");
            };
            (planes, received)
        })
        .unzip();
    for planes in &imports {
        read_planes(&gpu, planes);
    }
    drop(imports);
    gpu.wait();
    for received in receivers {
        assert_eq!(
            received.try_iter().collect::<Vec<_>>(),
            [LeaseEvent::Presented, LeaseEvent::Released]
        );
    }
}

/// Imports an RGBA buffer on `gpu` and checks its texels and the lease.
fn assert_rgba_imports(gpu: &mut Gpu) {
    let buffer = allocate(HardwareBufferFormat::R8G8B8A8_UNORM);
    write_rgba(&buffer);
    let (events, received) = mpsc::channel();
    let ImportedHardwareBuffer::Rgba(texture) = gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
        .expect("an RGBA buffer imports")
    else {
        panic!("an RGBA buffer imports as an RGBA texture");
    };
    assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(received.try_recv(), Ok(LeaseEvent::Presented));

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    assert_eq!(
        gpu.read_texels(&view, WIDTH, HEIGHT),
        expected_rgba(),
        "sampled texels"
    );
    assert_eq!(gpu.copy_texels(&texture), expected_rgba(), "copied texels");
    assert!(
        received.try_recv().is_err(),
        "the lease must not be released while the texture is alive"
    );

    drop(view);
    drop(texture);
    gpu.wait();
    assert_eq!(received.try_recv(), Ok(LeaseEvent::Released));
}

/// A `Y8Cb8Cr8_420` buffer imports as planes holding the codes the CPU wrote,
/// whichever way the driver describes it: as `NV12`, which the planes alias,
/// or — as the Mali-G715 driver does — as an external format, which the import
/// converts on the GPU.
#[test]
fn ycbcr_420_buffer_imports_as_planes_with_its_codes() {
    let mut gpu = Gpu::new();
    let buffer = allocate(HardwareBufferFormat::Y8Cb8Cr8_420);
    write_ycbcr(&buffer);
    let (events, received) = mpsc::channel();
    let ImportedHardwareBuffer::Ycbcr420(planes) = gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
        .expect("a Y8Cb8Cr8_420 buffer imports")
    else {
        panic!("a Y8Cb8Cr8_420 buffer imports as YCbCr planes");
    };
    assert_eq!(received.try_recv(), Ok(LeaseEvent::Presented));
    let converted = planes.luma.texture().format() == wgpu::TextureFormat::R8Unorm;
    if converted {
        // Nothing reads the buffer once the conversion has completed, so the
        // lease comes back while the planes are still alive.
        gpu.wait();
        assert_eq!(received.try_recv(), Ok(LeaseEvent::Released));
        assert_eq!(
            planes.chroma.texture().format(),
            wgpu::TextureFormat::Rg8Unorm
        );
    } else {
        assert_eq!(planes.luma.texture().format(), wgpu::TextureFormat::NV12);
    }
    read_planes(&gpu, &planes);

    drop(planes);
    gpu.wait();
    if !converted {
        assert_eq!(received.try_recv(), Ok(LeaseEvent::Released));
    }
    assert_eq!(received.try_recv(), Err(mpsc::TryRecvError::Disconnected));
}

/// A device opened without `TEXTURE_FORMAT_NV12` cannot alias an `NV12`
/// texture, so a `Y8Cb8Cr8_420` buffer converts on the GPU — through a
/// `VkSamplerYcbcrConversion` built for the defined
/// `G8_B8R8_2PLANE_420_UNORM` format when the driver maps it, or for its
/// external format when it does not — and the planes hold the stored codes
/// exactly.
#[test]
fn device_without_nv12_converts_ycbcr_420() {
    let mut gpu = Gpu::without_nv12();
    let buffer = allocate(HardwareBufferFormat::Y8Cb8Cr8_420);
    let format = driver_format(&gpu, &buffer);
    println!("driver reports {format:?} for the allocated Y8Cb8Cr8_420 buffer");
    write_ycbcr(&buffer);
    let (events, received) = mpsc::channel();
    let ImportedHardwareBuffer::Ycbcr420(planes) = gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
        .expect("a Y8Cb8Cr8_420 buffer imports without TEXTURE_FORMAT_NV12")
    else {
        panic!("a Y8Cb8Cr8_420 buffer imports as YCbCr planes");
    };
    // Without `TEXTURE_FORMAT_NV12` the buffer cannot alias, so the planes
    // must be the conversion's own textures, whichever way the driver
    // describes the buffer.
    assert_eq!(
        planes.luma.texture().format(),
        wgpu::TextureFormat::R8Unorm,
        "the buffer ({format:?}) must be converted, not aliased"
    );
    assert_eq!(
        planes.chroma.texture().format(),
        wgpu::TextureFormat::Rg8Unorm
    );
    read_planes(&gpu, &planes);
    drop(planes);
    gpu.wait();
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released]
    );
}

/// A binary semaphore the GPU signals after a long compute dispatch, exported
/// as the sync file a producer would hand over with its buffer.
struct ProducerFence {
    device: ash::Device,
    semaphore: vk::Semaphore,
}

impl ProducerFence {
    fn signal(gpu: &Gpu) -> (Self, OwnedFd) {
        // SAFETY: the test device is a Vulkan device; the raw device is used
        // only to create, export and destroy a semaphore of the test's own.
        let hal_device = unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() }
            .expect("the test device is Vulkan");
        let device = hal_device.raw_device().clone();
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        // SAFETY: a binary semaphore exportable as a sync file; the device
        // enables `VK_KHR_external_semaphore_fd` through `request_device`.
        let semaphore = unsafe { device.create_semaphore(&create_info, None) }
            .expect("failed to create the producer semaphore");
        let loader = ash::khr::external_semaphore_fd::Device::new(
            hal_device.shared_instance().raw_instance(),
            &device,
        );
        drop(hal_device);
        {
            // SAFETY: the test queue is Vulkan; the guard only stages a signal
            // of the semaphore on the next submission, which is the one below.
            let hal_queue = unsafe { gpu.queue.as_hal::<wgpu::hal::api::Vulkan>() }
                .expect("the test queue is Vulkan");
            hal_queue.add_signal_semaphore(semaphore, None);
        }
        gpu.spin();
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        // SAFETY: the semaphore's signal was just submitted, which a sync-file
        // export requires, and the extension is enabled.
        let fd = unsafe { loader.get_semaphore_fd(&get_info) }
            .expect("failed to export the producer semaphore");
        assert!(
            fd >= 0,
            "the spin dispatch finished before its semaphore was exported, so there is no \
             pending fence to test with"
        );
        // SAFETY: a successful export hands a new sync file to the caller.
        let fence = unsafe { OwnedFd::from_raw_fd(fd) };
        (Self { device, semaphore }, fence)
    }
}

impl Drop for ProducerFence {
    fn drop(&mut self) {
        // SAFETY: the test waits for the device to go idle before dropping
        // this, so the submission that signalled the semaphore has completed.
        unsafe { self.device.destroy_semaphore(self.semaphore, None) };
    }
}

/// The import takes a sync file that is still pending and turns it into the
/// semaphore its submission waits on, for an aliased RGBA buffer and for a
/// YCbCr buffer, which the device may convert. `wgpu` orders submissions on
/// one queue, so this proves the fence is consumed and the import completes
/// with the right contents, not that the wait alone orders the two.
#[test]
fn import_consumes_a_pending_acquire_fence() {
    let mut gpu = Gpu::new();
    let rgba = allocate(HardwareBufferFormat::R8G8B8A8_UNORM);
    write_rgba(&rgba);
    let (producer, fence) = ProducerFence::signal(&gpu);
    let (events, received) = mpsc::channel();
    let ImportedHardwareBuffer::Rgba(texture) = gpu
        .importer
        .import(frame(&rgba, Some(fence), RecordingLease(events)))
        .expect("an RGBA buffer imports with an acquire fence")
    else {
        panic!("an RGBA buffer imports as an RGBA texture");
    };
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    assert_eq!(gpu.read_texels(&view, WIDTH, HEIGHT), expected_rgba());
    drop(view);
    drop(texture);
    gpu.wait();
    drop(producer);
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released]
    );

    let ycbcr = allocate(HardwareBufferFormat::Y8Cb8Cr8_420);
    write_ycbcr(&ycbcr);
    let (producer, fence) = ProducerFence::signal(&gpu);
    let (events, received) = mpsc::channel();
    let ImportedHardwareBuffer::Ycbcr420(planes) = gpu
        .importer
        .import(frame(&ycbcr, Some(fence), RecordingLease(events)))
        .expect("a Y8Cb8Cr8_420 buffer imports with an acquire fence")
    else {
        panic!("a Y8Cb8Cr8_420 buffer imports as YCbCr planes");
    };
    read_planes(&gpu, &planes);
    drop(planes);
    gpu.wait();
    drop(producer);
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released]
    );
}

#[test]
fn rejected_buffer_drops_its_lease() {
    let mut gpu = Gpu::new();
    let buffer = HardwareBuffer::allocate(HardwareBufferDesc {
        width: WIDTH,
        height: HEIGHT,
        layers: 1,
        format: HardwareBufferFormat::R8G8B8A8_UNORM,
        usage: HardwareBufferUsage::CPU_WRITE_OFTEN | HardwareBufferUsage::CPU_READ_OFTEN,
        stride: 0,
    })
    .expect("failed to allocate a CPU-only hardware buffer");
    let (events, received) = mpsc::channel();
    let error = gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
        .expect_err("a buffer without GPU_SAMPLED_IMAGE is rejected");
    assert!(
        matches!(error, HardwareBufferImportError::NotGpuSampled(_)),
        "unexpected rejection: {error}"
    );
    // The lease was dropped with the frame, never presented or released.
    assert_eq!(received.try_recv(), Err(mpsc::TryRecvError::Disconnected));
}

/// A 10-bit buffer the driver describes only through an external format is
/// rejected rather than converted into 8-bit planes, which would drop
/// precision. A driver that maps `YCbCr_P010` to a Vulkan format rejects it as
/// unsupported instead.
#[test]
fn external_format_other_than_8_bit_ycbcr_is_rejected() {
    let mut gpu = Gpu::new();
    let buffer = allocate(HardwareBufferFormat::YCbCr_P010);
    let (events, received) = mpsc::channel();
    let error = gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
        .expect_err("a YCbCr_P010 buffer is rejected");
    match error {
        HardwareBufferImportError::ExternalFormat {
            format,
            external_format,
        } => {
            assert_eq!(format, HardwareBufferFormat::YCbCr_P010);
            assert_ne!(external_format, 0, "Vulkan external formats are never zero");
        }
        HardwareBufferImportError::UnsupportedFormat { format, .. } => {
            assert_eq!(format, HardwareBufferFormat::YCbCr_P010);
        }
        other => panic!("unexpected rejection of a YCbCr_P010 buffer: {other}"),
    }
    assert_eq!(received.try_recv(), Err(mpsc::TryRecvError::Disconnected));
}
