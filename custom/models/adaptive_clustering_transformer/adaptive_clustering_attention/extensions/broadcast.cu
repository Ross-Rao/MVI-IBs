#include <torch/extension.h>

typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float_3d;
typedef torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> float_2d;

template<typename GroupType>
__global__ void broadcast_kernel(
    const float_3d y,
    const torch::PackedTensorAccessor32<GroupType, 2, torch::RestrictPtrTraits> group,
    const float_2d weights,
    float_3d x
) {
    int B = x.size(0);
    int N = x.size(1);
    int D = x.size(2);
    int64_t C = y.size(1); // Use int64_t to handle large indices

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = idx / N;
    int n_idx = idx % N;
    if (b_idx >= B) return;

    GroupType c_idx = group[b_idx][n_idx];
    if ((std::is_signed<GroupType>::value && c_idx < 0) || c_idx >= C) return;

    float w = weights[b_idx][static_cast<int64_t>(c_idx)]; // Ensure correct type conversion
    for (int d_idx = 0; d_idx < D; d_idx++) {
        // x[b][n] belongs to only one cluster
        x[b_idx][n_idx][d_idx] = y[b_idx][c_idx][d_idx] * w;
    }
}

void broadcast(
    const torch::Tensor y,
    const torch::Tensor group,
    const torch::Tensor weights,
    torch::Tensor x
) {
    int B = x.size(0);
    int N = x.size(1);
    int D = x.size(2);

    const int threads = 1024;
    int blocks = (B * N + threads - 1) / threads;

    AT_DISPATCH_INTEGRAL_TYPES(group.scalar_type(), "broadcast", [&] {
        broadcast_kernel<scalar_t><<<blocks, threads>>>(
            y.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            group.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            x.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &broadcast);
}