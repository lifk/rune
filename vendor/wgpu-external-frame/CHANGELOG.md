# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/water-rs/wgpu-external-frame/compare/v0.3.0...v0.4.0) - 2026-10-07

### Added

- *(ahardware_buffer)* [**breaking**] convert defined-format YCbCr buffers when the device lacks NV12

### Other

- bound and retry every apt connection

## [0.3.0](https://github.com/water-rs/wgpu-external-frame/compare/v0.2.0...v0.3.0) - 2026-10-06

### Fixed

- *(ahardware_buffer)* [**breaking**] bind the YCbCr conversion from a descriptor pool

## [0.2.0](https://github.com/water-rs/wgpu-external-frame/compare/v0.1.0...v0.2.0) - 2026-10-05

### Added

- *(ahardware_buffer)* import Android hardware buffers through Vulkan
- *(io_surface)* [**breaking**] import on iOS and import each plane of 4:2:0 YCbCr surfaces
- *(deps)* upgrade wgpu from 29 to 30

### Fixed

- *(dma_buf)* split raw and wgpu commands across encoders
- *(ci)* make the MSRV job build with the declared floor ([#9](https://github.com/water-rs/wgpu-external-frame/pull/9))

### Other

- resolve rustdoc links on every target and deny rustdoc warnings in CI
- *(linux)* invoke vng from PATH, where pipx installs it on the runner
- *(linux)* run the tests inside a KVM guest
- *(dma_buf)* allocate the DMA-BUF on a vkms dumb buffer
- install the runner kernel's extra modules for udmabuf
- lint the Android build and link its device tests
- *(vulkan)* share memory-type selection between external-memory imports
- lint the Apple targets and build the iOS test binaries
- let the PR source gate accept release-plz release branches ([#8](https://github.com/water-rs/wgpu-external-frame/pull/8))
- disable incremental builds and trim debuginfo ([#7](https://github.com/water-rs/wgpu-external-frame/pull/7))
- run tests with cargo nextest ([#6](https://github.com/water-rs/wgpu-external-frame/pull/6))
- publish to crates.io via OIDC trusted publishing ([#2](https://github.com/water-rs/wgpu-external-frame/pull/2))
- gate pull requests into main so only dev may merge ([#3](https://github.com/water-rs/wgpu-external-frame/pull/3))
- update Linux package matrix and add dxc on Windows
