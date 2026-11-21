//! Extended-range (HDR) output: what the app may ask for, what the surface took, and the
//! mapping between the two.
//!
//! HDR here is **scRGB**: an `Rgba16Float` surface in the `ExtendedSrgbLinear` colour space,
//! drawn in linear light. A float target carries the whole linear range without a PQ pass, the
//! shaders write linear light, and the compositor owns gamut mapping, tone mapping and HDR
//! metadata. `Rgb10a2Unorm` + PQ is deliberately not chosen: every pass would have to encode PQ
//! before blending, and translucent UI over video would band at the seams.

/// Whether a window may present extended-range HDR output.
///
/// Set before `run` with `App::hdr`, or switch it from a handler with `Tx::set_hdr`. HDR needs a
/// surface that offers `Rgba16Float` in `ExtendedSrgbLinear`; where it doesn't, the window stays
/// on its SDR surface and [`hdr_status`] says so.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Hdr {
    /// Always present the SDR surface.
    #[default]
    Off,
    /// Present scRGB when the surface supports it, SDR otherwise.
    Auto,
}

/// What the last window to ask for a surface got.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HdrStatus {
    /// No window has configured its surface yet.
    #[default]
    Unknown,
    /// The window presents an SDR surface.
    Off,
    /// HDR was requested, but the surface offers no extended-range colour space.
    Unsupported,
    /// HDR was requested and the surface took the configuration but the driver refused it; the
    /// window fell back to SDR.
    Failed,
    /// The window presents extended-range linear (scRGB) output: 1.0 is 80 cd/m², and an SDR
    /// white is [`HDR_WHITE_SCALE`].
    Active {
        format: &'static str,
        color_space: &'static str,
    },
}

/// The last surface choice, for an app's settings and diagnostics. A window updates it when it
/// opens, when it is reconfigured, and when a driver refuses an HDR configuration.
static HDR_STATUS: std::sync::RwLock<HdrStatus> = std::sync::RwLock::new(HdrStatus::Unknown);

/// Reads [`HdrStatus`] without blocking on a writer for longer than the status write takes.
pub fn hdr_status() -> HdrStatus {
    *HDR_STATUS.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn set_hdr_status(status: HdrStatus) {
    let mut current = HDR_STATUS.write().unwrap_or_else(|poisoned| poisoned.into_inner());
    if *current != status {
        match status {
            HdrStatus::Active { format, color_space } => eprintln!("rune: HDR output on ({format}, {color_space})"),
            HdrStatus::Unsupported => eprintln!("rune: HDR was requested but this surface offers no extended-range colour space; using SDR"),
            HdrStatus::Failed => eprintln!("rune: HDR was requested but the driver refused the surface; using SDR"),
            HdrStatus::Off | HdrStatus::Unknown => {}
        }
        *current = status;
    }
}

/// SDR white in scRGB units: BT.2408 graphics white (203 cd/m²) over scRGB's 80 cd/m² unit. The
/// Wayland colour-management protocol suggests exactly this reference for Windows-scRGB, and
/// KWin maps SDR content against it; a UI drawn at 1.0 would be a dim 80 cd/m² otherwise.
pub const HDR_WHITE_SCALE: f32 = 203.0 / 80.0;

/// What a surface presents, and the target format the renderer draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SurfaceOutput {
    /// The surface carries extended-range linear colour (`ExtendedSrgbLinear`).
    pub hdr: bool,
    pub format: wgpu::TextureFormat,
    pub color_space: wgpu::SurfaceColorSpace,
}

impl SurfaceOutput {
    /// The renderer's target: an sRGB format is drawn into its non-sRGB twin when blending is in
    /// sRGB encoding, and a float HDR target keeps its own format.
    pub(crate) fn target(self, srgb_blend: bool) -> wgpu::TextureFormat {
        if srgb_blend { self.format.remove_srgb_suffix() } else { self.format }
    }
}

pub(crate) fn format_name(format: wgpu::TextureFormat) -> &'static str {
    match format {
        wgpu::TextureFormat::Rgba16Float => "Rgba16Float",
        wgpu::TextureFormat::Rgb10a2Unorm => "Rgb10a2Unorm",
        wgpu::TextureFormat::Bgra8Unorm => "Bgra8Unorm",
        wgpu::TextureFormat::Bgra8UnormSrgb => "Bgra8UnormSrgb",
        wgpu::TextureFormat::Rgba8Unorm => "Rgba8Unorm",
        wgpu::TextureFormat::Rgba8UnormSrgb => "Rgba8UnormSrgb",
        _ => "other",
    }
}

pub(crate) fn color_space_name(color_space: wgpu::SurfaceColorSpace) -> &'static str {
    match color_space {
        wgpu::SurfaceColorSpace::Auto => "Auto",
        wgpu::SurfaceColorSpace::Srgb => "Srgb",
        wgpu::SurfaceColorSpace::ExtendedSrgbLinear => "ExtendedSrgbLinear (scRGB)",
        wgpu::SurfaceColorSpace::ExtendedSrgb => "ExtendedSrgb",
        wgpu::SurfaceColorSpace::DisplayP3 => "DisplayP3",
        wgpu::SurfaceColorSpace::ExtendedDisplayP3 => "ExtendedDisplayP3",
        wgpu::SurfaceColorSpace::Bt2100Pq => "Bt2100Pq (HDR10)",
        wgpu::SurfaceColorSpace::Bt2100Hlg => "Bt2100Hlg",
    }
}

/// Picks the surface's format and colour space. HDR is `Rgba16Float` in scRGB; otherwise the
/// first sRGB format, or its non-sRGB twin when blending happens in sRGB encoding.
pub(crate) fn choose_output(caps: &wgpu::SurfaceCapabilities, srgb_blend: bool, hdr: Hdr) -> SurfaceOutput {
    if hdr != Hdr::Off
        && caps.color_spaces(wgpu::TextureFormat::Rgba16Float).contains(wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR)
    {
        return SurfaceOutput {
            hdr: true,
            format: wgpu::TextureFormat::Rgba16Float,
            color_space: wgpu::SurfaceColorSpace::ExtendedSrgbLinear,
        };
    }
    let fallback = caps.formats.first().copied().unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
    let srgb = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(fallback);
    let (format, color_space) = if srgb_blend {
        let non_srgb = srgb.remove_srgb_suffix();
        if caps.formats.contains(&non_srgb) {
            (non_srgb, wgpu::SurfaceColorSpace::Auto)
        } else {
            (srgb, wgpu::SurfaceColorSpace::Auto)
        }
    } else {
        (srgb, wgpu::SurfaceColorSpace::Auto)
    };
    SurfaceOutput { hdr: false, format, color_space }
}
