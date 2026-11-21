//! Window GPU surface configuration, pacing, backoff, presentation tracking and platform handles.

use crate::render::{Gpu, Renderer};
use crate::runtime::RuntimeCore;

pub(in crate::window) mod backoff;
pub(in crate::window) mod configure;
pub(in crate::window) mod corners;
#[cfg(target_os = "macos")]
pub(in crate::window) mod launch_drawables;
pub(in crate::window) mod loop_frame;
pub(in crate::window) mod native;
pub(in crate::window) mod pacing;
pub(in crate::window) mod presented;
pub(in crate::window) mod redraw;
pub(in crate::window) mod win;

pub(crate) use backoff::{Backoff, BackoffEffect, BackoffEvent};
pub(crate) use configure::configure;
#[cfg(target_os = "macos")]
pub(in crate::window) use launch_drawables::LaunchDrawables;
pub(in crate::window) use loop_frame::LoopFrame;
pub(in crate::window) use native::Native;
pub(crate) use pacing::{
    HiddenPace, Missed, TimerFrame, TIMER_FRAME, acquire, acquire_checked, timer_deadline_core,
    timer_wants_frame_core,
};
#[cfg(test)]
pub(crate) use pacing::{
    HIDDEN_REFRESH, OCCLUDED_MAX, OCCLUDED_RETRY, TIMEOUT_RETRY, TIMER_FRAME_STALL, timer_deadline,
    timer_wants_frame, wall_wake_deadline,
};
pub(crate) use presented::Presented;
#[cfg(test)]
pub(in crate::window) use presented::composing;
pub(in crate::window) use redraw::{Redraw, RedrawEffect, RedrawEvent};
pub(in crate::window) use win::{SurfaceRetries, SurfaceStep, SURFACE_TRIES, Win};

#[cfg(test)]
#[path = "surface/backoff_tests.rs"]
mod backoff_tests;
#[cfg(all(test, target_os = "macos"))]
#[path = "surface/occluded_surface_tests.rs"]
mod occluded_surface_tests;

/// Frames a window's swap chain may hold that the compositor hasn't shown. Windows: one, so a
/// frame shows at the first composition after it's done, and the window waits for room itself
/// (`slot`) rather than in `acquire`. Elsewhere two.
pub(crate) const FRAME_LATENCY: u32 = if cfg!(target_os = "windows") { 1 } else { 2 };

/// A vsynced sRGB (or non-sRGB when `srgb_blend` is set) configuration for the surface, and the
/// first alpha mode it supports that composites translucent pixels.
pub(crate) fn surface_config(surface: &wgpu::Surface, adapter: &wgpu::Adapter, (width, height): (u32, u32), srgb_blend: bool)
    -> Option<(wgpu::SurfaceConfiguration, Option<wgpu::CompositeAlphaMode>)> {
    surface_config_hdr(surface, adapter, (width, height), srgb_blend, crate::window::Hdr::Off)
        .map(|(config, alpha, _)| (config, alpha))
}

/// [`surface_config`] with this window's HDR preference. The returned output says whether the
/// surface is HDR, so the caller can put the runtime into linear blending and scale its white.
pub(crate) fn surface_config_hdr(surface: &wgpu::Surface, adapter: &wgpu::Adapter, (width, height): (u32, u32),
    srgb_blend: bool, hdr: crate::window::Hdr)
    -> Option<(wgpu::SurfaceConfiguration, Option<wgpu::CompositeAlphaMode>, crate::window::SurfaceOutput)> {
    let mut config = surface.get_default_config(adapter, width.max(1), height.max(1))?;
    let caps = surface.get_capabilities(adapter);
    let output = crate::window::choose_output(&caps, srgb_blend, hdr);
    config.format = output.format;
    config.color_space = output.color_space;
    if !output.hdr && srgb_blend {
        // Gamma blending into an sRGB format's non-sRGB twin, or an sRGB-encoded view of it.
        let non_srgb = output.format.remove_srgb_suffix();
        if non_srgb != output.format && !caps.formats.contains(&non_srgb) {
            config.view_formats = vec![non_srgb];
        }
    }
    // Backdrop copies read the drawable mid-frame (`ActivePass::split`: backdrop blur, effect
    // backdrops, rune-fx blend modes). On Metal it costs `framebufferOnly`, but without it they
    // quietly fall back.
    config.usage |= caps.usages & wgpu::TextureUsages::COPY_SRC;
    // A frame that redrew part of itself is copied in whole from the one the renderer keeps
    // (`render::damaged`).
    config.usage |= caps.usages & wgpu::TextureUsages::COPY_DST;
    config.present_mode = wgpu::PresentMode::AutoVsync;
    config.desired_maximum_frame_latency = FRAME_LATENCY;
    Some((config, crate::render::translucency::mode(surface, adapter), output))
}

/// Follows a blend switch (`Tx::set_srgb_blend`) on a configured surface: reconfigures it for
/// `rt`'s mode at its size and alpha mode, rebuilds `renderer` when its target format changes,
/// and rebuilds the `alpha` pass (when there is one) for the new format. Returns false when the
/// surface can't be configured.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn reblend_core(surface: &wgpu::Surface, gpu: &Gpu, config: &mut wgpu::SurfaceConfiguration,
    renderer: &mut Renderer, alpha: &mut Option<crate::render::surface_alpha::AlphaSurface>, rt: &mut RuntimeCore) -> bool {
    // The app's own blend mode picks the SDR format, so negotiate with it rather than with a
    // blend a previous HDR configuration forced.
    let preference = rt.blend_preference();
    let Some((next, _, output)) = surface_config_hdr(surface, &gpu.adapter, (config.width, config.height), preference, rt.hdr()) else { return false };
    if output.hdr {
        rt.force_srgb_blend(false);
        rt.set_output_scale(crate::window::HDR_WHITE_SCALE);
    } else {
        rt.force_srgb_blend(preference);
        rt.set_output_scale(1.0);
    }
    if (config.format, config.color_space, &config.view_formats) != (next.format, next.color_space, &next.view_formats) {
        (config.format, config.color_space, config.view_formats) = (next.format, next.color_space, next.view_formats);
        if let Err(error) = configure(surface, gpu, config) {
            eprintln!("rune: could not reconfigure a surface for a blend switch: {error}");
            crate::window::set_hdr_status(crate::window::HdrStatus::Failed);
            return false;
        }
    }
    let target = output.target(rt.srgb_blend());
    rebuild_renderer_core(&gpu.device, renderer, target, rt);
    if alpha.is_some() {
        *alpha = Some(crate::render::surface_alpha::AlphaSurface::new(&gpu.device, target, (config.width, config.height)));
    }
    crate::window::set_hdr_status(if output.hdr {
        crate::window::HdrStatus::Active {
            format: crate::window::format_name(output.format),
            color_space: crate::window::color_space_name(output.color_space),
        }
    } else if rt.hdr() == crate::window::Hdr::Off {
        crate::window::HdrStatus::Off
    } else {
        crate::window::HdrStatus::Unsupported
    });
    true
}

/// Switches `renderer` to `target` when its format changed, keeping its atlases, layer targets
/// and extra pipelines, and sets its blend mode to `rt`'s.
pub(crate) fn rebuild_renderer_core(device: &wgpu::Device, renderer: &mut Renderer, target: wgpu::TextureFormat, rt: &mut RuntimeCore) {
    renderer.set_format(device, target);
    renderer.set_srgb_blend(rt.srgb_blend());
    renderer.set_output_scale(rt.output_scale());
    // Before the first frame, so its glyph masks go to the GPU fill when the renderer has one.
    renderer.attach_text(&mut rt.text.atlas);
}
