#version 450

// Covers the render target with one triangle, corners at clip coordinates
// (-1, -1), (3, -1) and (-1, 3), drawn with three vertices and no vertex
// buffer. `uv` runs from 0 to 1 across the target, (0, 0) at its top-left
// corner, so each fragment receives the normalized coordinate of its own
// texel centre.

layout(location = 0) out vec2 uv;

void main() {
    uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
