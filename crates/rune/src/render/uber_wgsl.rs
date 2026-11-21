//! The uber shader's WGSL, Classic and Windfoil, joined when Rune builds (`build.rs`) with every
//! `\r\n` read as `\n`, and the BLAKE3 of each: the precompiled shaders are used when the hash
//! recorded beside them is this one, which the build knows, so a launch hashes nothing.

include!(concat!(env!("OUT_DIR"), "/uber_wgsl.rs"));

/// The uber fragment's last line. An HDR window's shader variant replaces it; the canonical WGSL
/// stays byte-for-byte what the precompiled Metal and DX12 libraries were built from, so only an
/// HDR renderer compiles its variant at run time.
pub(crate) const FINAL_RETURN: &str = "    return source;\n}";

/// The HDR window variant of an uber WGSL source: the window pass (`projection.z == 0`) scales
/// linear light by `g.flags.z` (SDR white in the surface's 80 cd/m² scRGB unit). An isolated
/// layer (`projection.z != 0`) keeps 1.0 = SDR white and is scaled when it is composited.
/// `None` when the fragment's final return moved, which a test catches.
pub(crate) fn hdr(base: &str) -> Option<String> {
    const SCALED: &str = "    // HDR (scRGB) window target: `g.flags.z` is SDR white in the surface's\n    // 80 cd/m² unit. An isolated layer keeps 1.0 = SDR white and is scaled when\n    // composited, so only the window pass applies it.\n    let scale = select(1.0, g.flags.z, projection.z == 0.0);\n    return vec4<f32>(source.rgb * scale, source.a);\n}";
    // A layer composite returns before the fragment's last line: it needs the same scale.
    const LAYER_RETURN: &str = "        return color * i.fill.a * clip_alpha(screen, i.clip, i.fade) * hole_alpha(screen, i.radii);";
    const LAYER_SCALED: &str = "        let composed = color * i.fill.a * clip_alpha(screen, i.clip, i.fade) * hole_alpha(screen, i.radii);\n        let layer_scale = select(1.0, g.flags.z, projection.z == 0.0);\n        return vec4<f32>(composed.rgb * layer_scale, composed.a);";
    if base.matches(FINAL_RETURN).count() != 1 || base.matches(LAYER_RETURN).count() != 1 {
        return None;
    }
    Some(base.replacen(FINAL_RETURN, SCALED, 1).replacen(LAYER_RETURN, LAYER_SCALED, 1))
}

/// Whether two strings are equal, for constants.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(super) const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() { return false; }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] { return false; }
        i += 1;
    }
    true
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hdr_variant_replaces_the_window_return_once() {
        let base = crate::render::variant::windfoil_wgsl();
        let out = hdr(base).expect("the uber fragment's final return moved");
        assert_eq!(out.matches(FINAL_RETURN).count(), 0);
        assert_eq!(
            out.matches("let scale = select(1.0, g.flags.z, projection.z == 0.0);")
                .count(),
            1
        );
        assert_eq!(out.matches("let layer_scale = select(1.0, g.flags.z, projection.z == 0.0);").count(), 1);
        assert_eq!(out.matches("return select(color, clamp(color, vec4(0.0), vec4(1.0)), i.params.x == 1.0)").count(), 0);
    }

    #[test]
    fn a_source_without_the_return_is_refused() {
        assert!(hdr("fn fs() {}").is_none());
        assert!(hdr(&format!("{FINAL_RETURN}{FINAL_RETURN}")).is_none(), "an ambiguous source is refused");
    }
}
