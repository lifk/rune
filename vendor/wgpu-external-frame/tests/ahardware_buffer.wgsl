// Reads every texel of a view back as packed 8-bit unorm channels, so a test
// can compare what the GPU samples from an imported buffer with what the CPU
// wrote into it.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> texels: array<u32>;

@compute @workgroup_size(8, 8)
fn read_texels(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(source);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    texels[id.y * size.x + id.x] = pack4x8unorm(textureLoad(source, id.xy, 0));
}
