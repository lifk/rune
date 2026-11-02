//! How YCbCr samples map to R'G'B', shared by every import that hands out
//! YCbCr planes.

/// How a YCbCr frame's samples map to non-linear R'G'B': the matrix and the
/// code range a consumer needs to convert them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct YcbcrEncoding {
    /// The YCbCr to R'G'B' matrix.
    pub matrix: YcbcrMatrix,
    /// The code range the samples span.
    pub range: YcbcrRange,
}

/// YCbCr to R'G'B' matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum YcbcrMatrix {
    /// ITU-R BT.601.
    Bt601,
    /// ITU-R BT.709.
    Bt709,
    /// ITU-R BT.2020, non-constant luminance.
    Bt2020,
}

/// The code range YCbCr samples span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum YcbcrRange {
    /// Video ("studio") range: luma spans 16–235 and chroma 16–240, at 8 bits.
    Video,
    /// Full range: luma and chroma span every code value.
    Full,
}
