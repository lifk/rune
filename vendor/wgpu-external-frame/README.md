# wgpu-external-frame

Import an externally produced GPU frame into [`wgpu`](https://wgpu.rs) as a
texture, without copying it through the CPU.

Browser engines, media decoders, screen capture, and compositors all publish
their output as a platform handle to memory the GPU already holds. This crate
turns each of those handles into a `wgpu::Texture` on a device you already own:

| Platform | Handle | Import path |
| --- | --- | --- |
| Linux | DMA-BUF file descriptor | Vulkan `VK_EXT_external_memory_dma_buf`, or EGL `EGL_LINUX_DMA_BUF_EXT` + `glEGLImageTargetTexture2DOES` |
| Android | `AHardwareBuffer` | Vulkan `VK_ANDROID_external_memory_android_hardware_buffer`; buffers of an external format are converted through a `VkSamplerYcbcrConversion` |
| macOS, iOS | `IOSurface` | `MTLDevice newTextureWithDescriptor:iosurface:plane:`, one texture per plane |
| Windows | Shared texture `HANDLE` | `ID3D12Device::OpenSharedHandle` |

Each platform is its own module with its own frame types — `DmaBufFrame`,
`HardwareBufferFrame`, `PackedIoSurfaceFrame` and `Ycbcr420IoSurfaceFrame`,
`SharedHandleFrame` — because the handles have nothing in common beyond the
goal. The Linux and Android sides additionally model the producer's *lease* on
the buffer (`DmaBufLease`, `HardwareBufferLease`), since those buffers usually
come from a pool the producer needs back, guarded by an explicit fence. An
`IOSurface` frame takes the object its producer recycles, such as a pooled
`CVPixelBuffer`, as its *owner* (`with_owner`), and drops it once the frame and
every texture imported from it have been destroyed, which `wgpu` does only
after every submission that used them has completed.

On Android an RGBA buffer imports as an `Rgba8Unorm` texture that aliases the
buffer itself, so nothing is copied. A 4:2:0 YCbCr buffer imports as two plane
views — luma `R8Unorm`, interleaved Cb/Cr `Rg8Unorm` — with the matrix and
range the Vulkan driver reports for it:

- when the driver maps the buffer to `G8_B8R8_2PLANE_420_UNORM` and the
  device has `TEXTURE_FORMAT_NV12`, the views are planes of one `NV12`
  texture that aliases the buffer;
- when the device lacks that feature, or the driver describes the buffer only
  with an implementation-defined external format — as some drivers do for
  every YCbCr buffer, camera frames included — sampling it needs a
  `VkSamplerYcbcrConversion`, which `wgpu` cannot express. The import then
  converts it on the GPU, in a small raw Vulkan pass on the importer's queue,
  into two textures `wgpu` owns. No pixel passes through the CPU.

The device must be opened with the extensions and the YCbCr conversion feature
the import needs, which `wgpu` does not enable on its own;
`ahardware_buffer::request_device` does that. The conversion needs nothing
more, so it runs on every device that can import at all, the Android emulator
included. It binds each buffer through a descriptor set from a pool of the
device's `maxCombinedImageSamplerDescriptorCount` descriptors where it reports
one (`VK_KHR_maintenance6`), and of 3, one per plane, otherwise; a driver that
needs more is reported as
`HardwareBufferImportError::ConversionDescriptorPool`. Building for Android
compiles the conversion's GLSL shaders with the NDK's `glslc`, found through
`ANDROID_NDK_HOME` or `ANDROID_NDK_ROOT`.

On Apple platforms, camera and video frames are biplanar 4:2:0 YCbCr surfaces,
and each of their planes imports as its own texture at that plane's extent:

| `IOSurface` pixel format | Frame type | Texture per plane | Device feature |
| --- | --- | --- | --- |
| packed BGRA or RGBA | `PackedIoSurfaceFrame` | `Bgra8Unorm` or `Rgba8Unorm` | — |
| `420v`, `420f` (8-bit 4:2:0) | `Ycbcr420IoSurfaceFrame` | luma `R8Unorm`, chroma `Rg8Unorm` | — |
| `x420`, `xf20` (10-bit 4:2:0) | `Ycbcr420IoSurfaceFrame` | luma `R16Unorm`, chroma `Rg16Unorm` | `TEXTURE_FORMAT_16BIT_NORM` |

Every import reaches through `wgpu` to its `hal` layer, so a device must be on
the backend its platform's import requires. Each entry point says which, and
asserts it.

## Status

This exists because `wgpu` has no portable external-memory import API yet. When
the work in the lineage of [wgpu#2320](https://github.com/gfx-rs/wgpu/issues/2320)
lands, most of what is here becomes a thin adapter over it, and the `hal`
reach-through goes away.