// Keeps the GPU busy long enough that work submitted after it is still pending
// when a test checks on it: a semaphore the Android tests export as a sync
// file, or a submission the IOSurface tests sample a texture in. One dispatch
// is kept short: on a Mali-G715 a dispatch ten times longer, run while other
// tests use the GPU, trips an "Internal firmware error" GPU reset that loses
// every device. A test that needs longer issues several dispatches.

@group(0) @binding(0) var<storage, read_write> sink: array<u32>;

@compute @workgroup_size(64)
fn spin(@builtin(global_invocation_id) id: vec3<u32>) {
    var state = id.x;
    for (var i = 0u; i < 100000u; i++) {
        state = state * 1664525u + 1013904223u;
    }
    sink[id.x] = state;
}
