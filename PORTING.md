# Porting the HDR patch onto the split workspace

State: branch `port/2026-11-21` off `upstream/dev`. `e7bfa9bd` is the pure-upstream baseline and
`a5c961c0` ports the symbol-fallback fix. This file lists what is still ours to move and where it
goes, so the work can resume without re-deriving it.

Reference: `/var/tmp/our-rune-patch.diff` (647 lines, our delta since the last merge) and
`/var/tmp/patches/*.diff` (the same patch split per file). Upstream has **no HDR**; every `Hdr`,
`HdrStatus`, `HDR_WHITE_SCALE`, `SurfaceOutput`, `output_scale` and `blend_preference` is ours.

## Applied cleanly by `git apply --3way`

`el/tx.rs`, `render.rs`, `render/metallib.rs`, `render/pipelines.rs`, `render/uber_wgsl.rs`,
`runtime/input/action.rs`, and the small ones re-anchored by hand: `lib.rs` exports,
`window/spec.rs` builder, `runtime/frame.rs` wiring, `runtime.rs` fields.

## Ported

`window/hdr.rs` (Hdr, HdrStatus, status, HDR_WHITE_SCALE, SurfaceOutput, choose_output),
`RuntimeCore` HDR state and methods (`hdr`, `set_hdr`, `blend_preference`, `force_srgb_blend`,
`output_scale`), the surface selection and reblend in `window/surface.rs`, the host's initial
configuration and `apply_hdr_output`, the app field/builder through `RuntimeConfig`, the uber
HDR variant selector (`render/variant.rs`), the globals scale (`render/upload.rs`), and the
surface tests. `cargo test -p rune --lib` passes (1826 tests at 4 threads).

## Still to port

1. **`crates/rune/src/window/hdr.rs` (new).** Move our `Hdr`, `HdrStatus`, `hdr_status`,
   `set_hdr_status`, `HDR_WHITE_SCALE`, `SurfaceOutput`, `format_name`, `color_space_name`,
   `choose_surface`/`reblend` out of the old `window.rs` diff into this module, and re-export from
   `window.rs`. Upstream's `window.rs` is now small; adding a sibling module avoids rewriting it.
2. **`runtime/paint/blend.rs`.** Our 52-line change: `blend_preference`, HDR forcing linear light,
   `set_hdr` (sets `hdr`, `output_scale`, blend, requests a reconfigure), and the linear composite.
   Upstream's file still has `set_srgb_blend`/`srgb_blend` delegating to `self.paint`; find the
   matching hook.
3. **`window/host/create.rs`** (was `window/create.rs`): pick the surface format/colour space with
   `hdr::choose_surface`, write `set_hdr_status`, set `rt.set_hdr`, and on a refused configuration
   fall back to SDR and set `HdrStatus::Failed`.
4. **`app.rs`**: our three hunks — `hdr` field, `App::hdr` builder (upstream already has
   `modifier_keys`/`tab_visits_controls`/`srgb_blend`/`font_smoothing`/`dev_stats` next to it), and
   passing `app.hdr` into `Host` in `window.rs`.
5. **`render/variant.rs`, `render/upload.rs`**: HDR target formats (Rgba16Float) in the uber variant
   and the upload path (the other render files already applied).
6. **`rune-media/src/native_surface/video.wgsl` and `pipeline.rs`**: our absolute PQ/HLG video math.
   `video.wgsl` exists upstream without it.
7. **`window/tests.rs`**: our 48 lines of HDR surface tests, adapted to the new host names.
8. **dmabuf**: our `dmabuf.rs` + `wgpu-external-frame` are deliberately absent here. The app's
   player stays on the old pin until the media port, which should target upstream's
   `push_dmabuf` (BGRA8 linear only) rather than resurrecting the vendor.

## Then, in Runefin

Pin the port branch, fix `player.rs` against the new `NativeSurface`, run `scripts/check.sh`, make
`scripts/snapshots.sh` produce identical PNGs (except the About row), and re-verify HDR on
RADV/Wayland.
