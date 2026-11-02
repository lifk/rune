use std::os::fd::OwnedFd;

use ndk::hardware_buffer::{HardwareBuffer, HardwareBufferDesc, HardwareBufferRef};
use ndk::hardware_buffer_format::HardwareBufferFormat;

/// The producer's ownership of the buffer behind a [`HardwareBufferFrame`].
///
/// A producer such as `AImageReader` hands out buffers from a fixed pool and
/// needs each one back — for `AImageReader`, by deleting the `AImage` that
/// carries it. That is a two-step protocol: the buffer is imported, then the
/// GPU stops reading it. This trait is both steps.
///
/// A lease is dropped rather than released whenever an import is abandoned
/// before it reaches the GPU, so implementations must treat `Drop` as an
/// implicit release.
pub trait HardwareBufferLease: core::fmt::Debug + Send {
    /// Tells the producer the buffer has been imported.
    ///
    /// The GPU may still be reading the buffer; only [`Self::release`] says it
    /// has finished.
    fn presented(&mut self);

    /// Returns the buffer to the producer.
    ///
    /// Called once the GPU no longer reads the buffer, so the producer may
    /// reuse it at once; there is no release fence to wait on. For a texture
    /// that aliases the buffer, that is once the texture has been dropped and
    /// every submission that used it has completed; for planes converted from
    /// an external format, once the conversion has completed. `wgpu` reports
    /// both lazily, so this runs on whichever thread drives the device at that
    /// moment — inside `wgpu::Device::poll` or `wgpu::Queue::submit` — and
    /// must not block on the GPU.
    fn release(self: Box<Self>);
}

/// A strong reference on an `AHardwareBuffer` that may move between threads.
pub(super) struct BufferReference(HardwareBufferRef);

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the one field is a reference on a thread-safe NDK object, as the safety comment argues"
)]
// SAFETY: `HardwareBufferRef` is not `Send` only because it wraps a raw
// pointer. An `AHardwareBuffer` is a reference-counted NDK object whose
// acquire, release, and describe entry points are thread-safe: the NDK
// documents hardware buffers as shareable across threads and processes, and
// the reference count is atomic. This type only ever releases its reference
// (on drop) and hands out the pointer to Vulkan, so moving it to another
// thread — the one on which `wgpu` destroys the imported texture — is sound.
unsafe impl Send for BufferReference {}

impl BufferReference {
    pub(super) fn buffer(&self) -> &HardwareBuffer {
        &self.0
    }
}

impl core::fmt::Debug for BufferReference {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_tuple("BufferReference")
            .field(&self.0.as_ptr())
            .finish()
    }
}

/// An `AHardwareBuffer` handed over for import, with the producer's acquire
/// fence and lease.
///
/// The frame holds its own reference on the buffer, taken at construction, so
/// the buffer stays valid for as long as the GPU reads it.
#[derive(Debug)]
pub struct HardwareBufferFrame {
    buffer: BufferReference,
    description: HardwareBufferDesc,
    acquire_fence: Option<OwnedFd>,
    lease: Option<Box<dyn HardwareBufferLease>>,
}

/// The parts of a frame an import consumes.
pub(super) struct FrameParts {
    pub(super) buffer: BufferReference,
    pub(super) description: HardwareBufferDesc,
    pub(super) acquire_fence: Option<OwnedFd>,
    pub(super) lease: Option<Box<dyn HardwareBufferLease>>,
}

impl HardwareBufferFrame {
    /// Takes a reference on `buffer`, so that it stays valid past the callback
    /// or `AImage` that handed it over.
    ///
    /// `acquire_fence` is the producer's sync file that signals once it has
    /// finished writing the buffer, such as the one
    /// `AImageReader_acquireNextImageAsync` returns; `None` means the buffer
    /// is ready as it arrives. The import makes the GPU wait on the fence; the
    /// CPU never does.
    ///
    /// Attach the producer's lease with [`Self::with_lease`].
    #[must_use]
    pub fn new(buffer: &HardwareBuffer, acquire_fence: Option<OwnedFd>) -> Self {
        let buffer = buffer.acquire();
        let description = buffer.describe();
        Self {
            buffer: BufferReference(buffer),
            description,
            acquire_fence,
            lease: None,
        }
    }

    /// Attaches the producer's lease on the buffer this frame borrows.
    ///
    /// # Panics
    ///
    /// Panics when the frame already carries a lease, since only one producer
    /// can own the buffer.
    #[must_use]
    pub fn with_lease(mut self, lease: Box<dyn HardwareBufferLease>) -> Self {
        assert!(
            self.lease.is_none(),
            "an AHardwareBuffer frame carries at most one buffer lease"
        );
        self.lease = Some(lease);
        self
    }

    /// The buffer this frame holds a reference on.
    #[must_use]
    pub fn buffer(&self) -> &HardwareBuffer {
        self.buffer.buffer()
    }

    /// The buffer's description, read once when the frame was created.
    #[must_use]
    pub const fn description(&self) -> HardwareBufferDesc {
        self.description
    }

    /// Whether a consumer must ignore the imported alpha channel and treat the
    /// frame as opaque.
    ///
    /// `AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM` imports as an RGBA texture whose
    /// alpha is undefined once an external producer has written it.
    #[must_use]
    pub fn force_opaque(&self) -> bool {
        self.description.format == HardwareBufferFormat::R8G8B8X8_UNORM
    }

    pub(super) fn into_parts(self) -> FrameParts {
        FrameParts {
            buffer: self.buffer,
            description: self.description,
            acquire_fence: self.acquire_fence,
            lease: self.lease,
        }
    }
}
