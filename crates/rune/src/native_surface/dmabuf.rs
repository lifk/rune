//! Linux DMA-BUF frames: a decoder's own buffers, imported as textures without a copy.
//!
//! A VA-API decoder hands out its pictures as file descriptors (GStreamer's
//! `video/x-raw(memory:DMABuf)`). The pixels are already in GPU memory, so a 4K frame costs an
//! import, not ~25 MB of copying — which is the whole reason this exists instead of `ByteFrame`.
//!
//! The format and the DRM modifier are the producer's, not ours, and neither is negotiable: a
//! modifier describes tiling the sampler cannot guess. A frame that does not pass [`DmabufFrame::new`]
//! is rejected before it can reach the renderer, because a wrong stride shows as diagonal shearing
//! that looks exactly like a driver bug.

use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::sync::Mutex;

use wgpu_external_frame::dma_buf::{
    DRM_FORMAT_MOD_INVALID, DmaBufFormat, DmaBufFrame as ImportFrame, DmaBufImport,
    DmaBufImporter, DmaBufPlane,
};

use super::{ColorInfo, Format};

/// The producer's ownership of a buffer, released once the GPU is done reading it.
pub use wgpu_external_frame::dma_buf::DmaBufLease as Lease;

/// One plane of a DMA-BUF: where the pixels start inside `fd`, in bytes.
#[derive(Debug)]
pub struct Plane {
    pub fd: OwnedFd,
    /// Byte offset of the first pixel in the buffer.
    pub offset: u64,
    /// Bytes per row, padding included.
    pub stride: u64,
}

/// A frame whose memory the producer owns until this value is dropped.
///
/// Dropping it does not itself tell the producer anything: the consumer (the renderer) drops its
/// [`Plane`]s once the last submission that sampled them has completed, and *that* is the signal
/// the decoder's pool waits for. Releasing early is how a buffer gets refilled mid-draw.
#[derive(Debug)]
pub struct DmabufFrame {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub color: ColorInfo,
    /// DRM format modifier; `DRM_FORMAT_MOD_LINEAR` is `0`.
    pub modifier: u64,
    pub planes: Vec<Plane>,
    /// Moved into the imported frame, which releases it after the copy completes.
    lease: Mutex<Option<Box<dyn Lease>>>,
}

impl Plane {
    /// Duplicates a producer-owned descriptor, so the frame owns its fd and the producer can
    /// close or recycle its original as soon as it hands the buffer over.
    pub fn from_raw_fd(fd: RawFd, offset: u64, stride: u64) -> Result<Plane, String> {
        // SAFETY: `libc::dup` only reads the descriptor table and returns a new owned descriptor.
        let duplicated = unsafe { libc::dup(fd) };
        if duplicated < 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        // SAFETY: `dup` succeeded, so this is a fresh valid descriptor owned by this value.
        let fd = unsafe { OwnedFd::from_raw_fd(duplicated) };
        Ok(Plane { fd, offset, stride })
    }
}

impl DmabufFrame {
    /// Checks the frame against everything the sampler needs, then hands it back.
    pub fn new(
        width: u32,
        height: u32,
        format: Format,
        color: ColorInfo,
        modifier: u64,
        planes: Vec<Plane>,
    ) -> Result<DmabufFrame, String> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err("video size must be 1..16384".into());
        }
        if format != Format::Bgra && (!width.is_multiple_of(2) || !height.is_multiple_of(2)) {
            return Err("NV12 requires even dimensions".into());
        }
        let wanted = Self::plane_count(format);
        if planes.len() != wanted {
            return Err(format!(
                "{} needs {wanted} plane(s), got {}",
                Self::format_name(format),
                planes.len()
            ));
        }
        for (i, plane) in planes.iter().enumerate() {
            // A 4:2:0 chroma plane is half the row width; the luma plane and a packed surface are
            // the full width. Anything narrower cannot hold the frame.
            let min = if format == Format::Bgra || i == 0 {
                u64::from(width) * Self::bytes_per_pixel(format)
            } else {
                u64::from(width)
            };
            if plane.stride < min {
                return Err(format!(
                    "plane {i} stride {} is under the {min} bytes a {}-wide row needs",
                    plane.stride, width
                ));
            }
        }
        Ok(DmabufFrame {
            width,
            height,
            format,
            color,
            modifier,
            planes,
            lease: Mutex::new(None),
        })
    }

    /// Attaches the producer's lease. It is released after the GPU has copied the frame, not
    /// when the decoder hands it over — that is what keeps a pooled buffer from being refilled
    /// while the shader is still reading it.
    pub fn with_lease(mut self, lease: Box<dyn Lease>) -> Self {
        self.lease = Mutex::new(Some(lease));
        self
    }

    /// The frame in the importer's shape.
    ///
    /// Only packed BGRA is accepted: `wgpu-external-frame` imports one 32-bit plane, and an
    /// NV12 buffer needs two images with a multi-plane modifier. Export BGRA from
    /// `vaapipostproc` until the multi-plane path lands; handing NV12 here is a visible error,
    /// never a sheared picture.
    pub(crate) fn to_external(&self) -> Result<ImportFrame, String> {
        if self.format != Format::Bgra {
            return Err("DMA-BUF import takes BGRA; NV12 needs the multi-plane path".into());
        }
        if self.modifier == DRM_FORMAT_MOD_INVALID {
            return Err("DMA-BUF has no negotiated DRM modifier".into());
        }
        let plane = self.planes.first().ok_or("BGRA frame has no plane")?;
        let fd = plane.fd.try_clone().map_err(|e| format!("duplicate DMA-BUF plane: {e}"))?;
        let offset = u32::try_from(plane.offset).map_err(|_| "DMA-BUF plane offset exceeds u32")?;
        let stride = u32::try_from(plane.stride).map_err(|_| "DMA-BUF stride exceeds u32")?;
        let frame = ImportFrame::new(
            self.width,
            self.height,
            DmaBufFormat::Bgra8,
            self.modifier,
            vec![DmaBufPlane { fd, offset, stride }],
            None,
        );
        let lease = self.lease.lock().map_err(|_| "DMA-BUF lease lock poisoned")?.take();
        Ok(match lease {
            Some(lease) => frame.with_lease(lease),
            None => frame,
        })
    }

    /// The `wgpu`/Rune format a DRM fourcc maps to, or `None` when we cannot sample it yet.
    ///
    /// P010 (10-bit HDR) is deliberately absent: it arrives with the tone-mapping work, and
    /// pretending to support it would show a green or sheared picture instead of an error.
    pub fn from_fourcc(code: u32) -> Option<Format> {
        match code {
            DRM_FORMAT_XRGB8888 | DRM_FORMAT_ARGB8888 => Some(Format::Bgra),
            DRM_FORMAT_NV12 => Some(Format::Nv12Video),
            _ => None,
        }
    }

    pub fn plane_count(format: Format) -> usize {
        match format {
            Format::Bgra => 1,
            Format::Nv12Video | Format::Nv12Full => 2,
        }
    }

    fn bytes_per_pixel(format: Format) -> u64 {
        match format {
            Format::Bgra => 4,
            Format::Nv12Video | Format::Nv12Full => 1,
        }
    }

    fn format_name(format: Format) -> &'static str {
        match format {
            Format::Bgra => "BGRA",
            Format::Nv12Video | Format::Nv12Full => "NV12",
        }
    }
}

/// Imports one frame into a texture wgpu owns, then releases the producer.
///
/// The copy runs on the GPU: publish this texture to the renderer instead of sampling the
/// producer's buffer directly, and the decoder may refill its pool as soon as the copy
/// completes. That is one blit more than a direct import, and it is the price of not
/// hand-writing the Vulkan acquire/release barriers around an external image.
pub(crate) struct Importer {
    inner: DmaBufImporter,
}

impl Importer {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
    ) -> Result<Self, String> {
        let backend = adapter.get_info().backend;
        if backend == wgpu::Backend::Vulkan
            && !device
                .features()
                .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
        {
            return Err(
                "this Vulkan device cannot import DMA-BUF frames (VK_EXT_external_memory_dma_buf \
                 is off); use the CPU frame path or update Mesa"
                    .into(),
            );
        }
        if !matches!(backend, wgpu::Backend::Vulkan | wgpu::Backend::Gl) {
            return Err(format!(
                "DMA-BUF import needs a Vulkan or GLES device, not {backend:?}"
            ));
        }
        Ok(Self {
            inner: DmaBufImporter::new(device, queue, adapter),
        })
    }

    /// Copies a frame into a wgpu-owned texture and returns the two planes the shader samples.
    ///
    /// The producer's lease is released in `on_submitted_work_done`, not here: the decoder
    /// must not recycle the buffer before the blit that reads it has finished.
    pub(crate) fn import(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &DmabufFrame,
    ) -> Result<[wgpu::Texture; 2], String> {
        let mut external = frame.to_external()?;
        let (width, height) = external.visible_size();
        let destination = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DMA-BUF frame"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: external.format.texture_format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let copy: DmaBufImport = self.inner.copy_into(&mut external, &destination);
        let guard = copy.guard;
        let mut command_buffers = copy.command_buffers;
        command_buffers.push(copy.encoder.finish());
        external.presented();
        queue.submit(command_buffers);
        queue.on_submitted_work_done(move || {
            // The import guard destroys the external image, and only now may the pool reuse it.
            drop(guard);
            external.release(None);
        });
        Ok([destination.clone(), destination])
    }
}

/// `fourcc_code('X','R','2','4')` from `drm_fourcc.h`, in little-endian byte order.
const fn fourcc(code: [u8; 4]) -> u32 {
    (code[0] as u32) | ((code[1] as u32) << 8) | ((code[2] as u32) << 16) | ((code[3] as u32) << 24)
}

pub const DRM_FORMAT_XRGB8888: u32 = fourcc(*b"XR24");
pub const DRM_FORMAT_ARGB8888: u32 = fourcc(*b"AR24");
pub const DRM_FORMAT_NV12: u32 = fourcc(*b"NV12");
/// Advertised so the error message can name it; not imported yet.
pub const DRM_FORMAT_P010: u32 = fourcc(*b"P010");

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::{AsFd, AsRawFd};

    fn plane(stride: u64) -> Plane {
        // A real plane owns a dmabuf fd; for validation any open fd will do.
        let file = std::fs::File::open("/dev/null").expect("open /dev/null");
        Plane {
            fd: file.into(),
            offset: 0,
            stride,
        }
    }

    fn frame(w: u32, h: u32, format: Format, strides: &[u64]) -> Result<DmabufFrame, String> {
        DmabufFrame::new(
            w,
            h,
            format,
            ColorInfo::default(),
            0,
            strides.iter().map(|s| plane(*s)).collect(),
        )
    }

    #[test]
    fn from_raw_fd_duplicates_a_live_descriptor() {
        let file = std::fs::File::open("/dev/null").expect("open /dev/null");
        let plane = Plane::from_raw_fd(file.as_raw_fd(), 0, 4).expect("dup");
        assert!(plane.fd.as_fd().as_raw_fd() >= 0);
        assert!(Plane::from_raw_fd(-1, 0, 4).is_err());
    }

    #[test]
    fn the_fourcc_codes_match_drm_fourcc_h() {
        assert_eq!(DRM_FORMAT_XRGB8888, 0x3432_5258);
        assert_eq!(DRM_FORMAT_ARGB8888, 0x3432_5241);
        assert_eq!(DRM_FORMAT_NV12, 0x3231_564e);
        assert_eq!(DRM_FORMAT_P010, 0x3031_3050);
    }

    #[test]
    fn a_packed_frame_maps_to_bgra_and_a_semi_planar_one_to_nv12() {
        assert_eq!(DmabufFrame::from_fourcc(DRM_FORMAT_XRGB8888), Some(Format::Bgra));
        assert_eq!(DmabufFrame::from_fourcc(DRM_FORMAT_NV12), Some(Format::Nv12Video));
    }

    #[test]
    fn ten_bit_is_refused_rather_than_sampled_wrong() {
        assert_eq!(DmabufFrame::from_fourcc(DRM_FORMAT_P010), None);
        assert_eq!(DmabufFrame::from_fourcc(fourcc(*b"RG24")), None);
    }

    #[test]
    fn a_well_formed_nv12_frame_is_accepted() {
        let f = frame(1920, 1080, Format::Nv12Video, &[1920, 1920]).expect("valid");
        assert_eq!(f.planes.len(), 2);
        assert_eq!(f.width, 1920);
    }

    #[test]
    fn odd_dimensions_are_refused_for_every_420_format() {
        for format in [Format::Nv12Video, Format::Nv12Full] {
            let err = frame(1919, 1080, format, &[1920, 1920]).unwrap_err();
            assert!(err.contains("even"), "{err}");
        }
    }

    #[test]
    fn a_packed_frame_may_have_an_odd_size() {
        // Nothing about BGRA needs even edges; refusing it would reject a legitimate crop.
        assert!(frame(1919, 1079, Format::Bgra, &[1919 * 4]).is_ok());
    }

    #[test]
    fn the_plane_count_is_enforced_in_both_directions() {
        assert!(frame(640, 480, Format::Nv12Video, &[640]).unwrap_err().contains("2 plane"));
        assert!(frame(640, 480, Format::Bgra, &[2560, 2560]).unwrap_err().contains("1 plane"));
    }

    #[test]
    fn a_stride_narrower_than_the_row_is_refused_with_the_numbers() {
        let err = frame(1920, 1080, Format::Nv12Video, &[1900, 1920]).unwrap_err();
        assert!(err.contains("plane 0") && err.contains("1900") && err.contains("1920"), "{err}");
    }

    #[test]
    fn padded_strides_are_fine_because_decoders_align_rows() {
        // 1088 is the classic aligned height, and rows are padded to 64 bytes or more.
        assert!(frame(1920, 1080, Format::Nv12Video, &[2048, 2048]).is_ok());
    }

    #[test]
    fn a_zero_or_absurd_size_is_refused() {
        assert!(frame(0, 1080, Format::Nv12Video, &[0, 0]).is_err());
        assert!(frame(20000, 1080, Format::Bgra, &[80000]).is_err());
    }

    #[test]
    fn the_plane_owns_its_descriptor() {
        let f = frame(640, 480, Format::Nv12Video, &[640, 640]).expect("valid");
        // If Plane held a borrowed fd this would not compile; the assertion is the type itself.
        let _borrowed: &dyn AsFd = &f.planes[0].fd;
    }

    #[test]
    fn packed_bgra_maps_to_the_importers_single_plane_format() {
        let f = frame(640, 480, Format::Bgra, &[2560]).expect("valid");
        let external = f.to_external().expect("packed frames import");
        assert_eq!(external.format, DmaBufFormat::Bgra8);
        assert_eq!(external.visible_size(), (640, 480));
        assert_eq!(external.planes.len(), 1);
    }

    #[test]
    fn nv12_is_refused_with_a_reason_instead_of_shearing() {
        let f = frame(640, 480, Format::Nv12Video, &[640, 640]).expect("valid");
        let error = f.to_external().unwrap_err();
        assert!(error.contains("BGRA") && error.contains("NV12"), "{error}");
    }

    #[test]
    fn a_missing_modifier_is_refused_before_vulkan_sees_it() {
        let f = DmabufFrame::new(
            640,
            480,
            Format::Bgra,
            ColorInfo::default(),
            DRM_FORMAT_MOD_INVALID,
            vec![plane(2560)],
        )
        .expect("valid");
        assert!(f.to_external().is_err());
    }
}
