#version 450

// Writes one plane of an external-format 4:2:0 YCbCr buffer into a texture.
//
// `frame` is a combined image sampler whose immutable sampler carries a
// `VkSamplerYcbcrConversion` with the `RGB_IDENTITY` model, so sampling
// returns the stored codes untouched, in the component order of a
// `G8_B8R8` format: red is Cr, green is Y', blue is Cb. The sampler filters
// with `NEAREST`, chroma reconstruction included.
//
// The luma pipeline renders at the frame's extent, so every fragment samples
// the centre of one luma texel. The chroma pipeline renders at half the
// extent, so every fragment samples the centre of one chroma texel, and the
// nearest chroma sample is that texel's own Cb/Cr pair whether the samples
// sit midway between luma texels or on the even ones.

// Which plane this pipeline writes: false for luma, true for chroma.
layout(constant_id = 0) const bool CHROMA = false;

layout(set = 0, binding = 0) uniform sampler2D frame;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 plane;

void main() {
    vec4 ycbcr = textureLod(frame, uv, 0.0);
    plane = CHROMA ? vec4(ycbcr.b, ycbcr.r, 0.0, 1.0) : vec4(ycbcr.g, 0.0, 0.0, 1.0);
}
