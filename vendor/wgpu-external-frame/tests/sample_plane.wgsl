// Loads every texel of one imported plane into a storage buffer, row-major,
// so the test reads back what a shader binding the plane actually sees.

@group(0) @binding(0) var plane: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> texels: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(plane);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    texels[id.y * size.x + id.x] = textureLoad(plane, id.xy, 0);
}
