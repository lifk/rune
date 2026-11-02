//! Compiles the Android YCbCr conversion shaders to SPIR-V.
//!
//! The shaders sample a combined image sampler that carries a
//! `VkSamplerYcbcrConversion`, which only a true `OpTypeSampledImage`
//! variable may do; `naga` cannot express one. They are GLSL, compiled with
//! the `glslc` the Android NDK ships in `shader-tools/`, the toolchain every
//! Android build of this crate already requires. Other targets compile no
//! shader.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The shader sources, relative to the package root. Each compiles to
/// `<file name>.spv` in `OUT_DIR`.
const SHADERS: [&str; 2] = [
    "src/ahardware_buffer/ycbcr_planes.vert",
    "src/ahardware_buffer/ycbcr_planes.frag",
];

/// The environment variables that name the NDK root, in the order they are
/// consulted.
const NDK_VARIABLES: [&str; 2] = ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("cargo sets CARGO_CFG_TARGET_OS");
    if target_os != "android" {
        return;
    }
    for variable in NDK_VARIABLES {
        println!("cargo::rerun-if-env-changed={variable}");
    }
    let glslc = ndk_glslc();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    for shader in SHADERS {
        println!("cargo::rerun-if-changed={shader}");
        compile(&glslc, Path::new(shader), &out_dir);
    }
}

/// Finds `glslc` in the NDK named by [`NDK_VARIABLES`].
fn ndk_glslc() -> PathBuf {
    let (variable, root) = NDK_VARIABLES
        .into_iter()
        .find_map(|variable| {
            env::var_os(variable)
                .filter(|value| !value.is_empty())
                .map(|value| (variable, PathBuf::from(value)))
        })
        .unwrap_or_else(|| {
            panic!(
                "building for Android compiles the YCbCr conversion shaders with the NDK's glslc; \
                 set {} to the root of the NDK this build uses",
                NDK_VARIABLES.join(" or ")
            )
        });
    let shader_tools = root.join("shader-tools");
    let executable: OsString = format!("glslc{}", env::consts::EXE_SUFFIX).into();
    let entries = fs::read_dir(&shader_tools).unwrap_or_else(|error| {
        panic!(
            "{variable} names {}, which has no readable shader-tools directory: {error}",
            root.display()
        )
    });
    // `shader-tools` holds one directory per host the NDK was built for, and
    // an NDK install carries exactly one.
    let candidates: Vec<PathBuf> = entries
        .map(|entry| {
            entry
                .unwrap_or_else(|error| {
                    panic!("failed to list {}: {error}", shader_tools.display())
                })
                .path()
                .join(&executable)
        })
        .filter(|candidate| candidate.is_file())
        .collect();
    match candidates.as_slice() {
        [glslc] => glslc.clone(),
        [] => panic!(
            "{variable} names {}, whose shader-tools directory holds no {}",
            root.display(),
            executable.display()
        ),
        several => panic!(
            "{variable} names {}, whose shader-tools directory holds several glslc executables: \
             {several:?}",
            root.display()
        ),
    }
}

fn compile(glslc: &Path, source: &Path, out_dir: &Path) {
    let file_name = source
        .file_name()
        .expect("every shader source names a file")
        .to_str()
        .expect("every shader file name is UTF-8");
    let output = out_dir.join(format!("{file_name}.spv"));
    let result = Command::new(glslc)
        .arg("--target-env=vulkan1.1")
        .arg("-O")
        .arg("-Werror")
        .arg("-o")
        .arg(&output)
        .arg(source)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", glslc.display()));
    assert!(
        result.status.success(),
        "{} failed to compile {} ({}):\n{}",
        glslc.display(),
        source.display(),
        result.status,
        String::from_utf8_lossy(&result.stderr)
    );
}
