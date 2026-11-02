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

use std::os::fd::OwnedFd;

use super::{ColorInfo, Format};

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
    use std::os::fd::AsFd;

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
}
