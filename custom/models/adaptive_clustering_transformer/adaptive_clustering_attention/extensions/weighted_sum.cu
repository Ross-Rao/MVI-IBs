#include <torch/extension.h>
#include <type_traits>  // 添加类型特性支持

typedef torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> float_2d;
typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float_3d;

template <typename GroupType>
__global__ void weighted_sum_kernel(
    const float_3d x,
    const torch::PackedTensorAccessor32<GroupType, 2, torch::RestrictPtrTraits> group,
    const float_2d weights,
    float_3d y) {
    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    const int64_t C = y.size(1);  // 使用 int64_t 防止大索引溢出

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b_idx = idx / N;
    const int n_idx = idx % N;
    if (b_idx >= B) return;

    const GroupType c_idx = group[b_idx][n_idx];

    // 动态检查负数（仅当 GroupType 是有符号类型时）
    if ((std::is_signed<GroupType>::value && c_idx < 0) ||
        (c_idx >= C)) return;

    const float w = weights[b_idx][static_cast<int64_t>(c_idx)];  // 显式类型转换
    for (int d_idx = 0; d_idx < D; d_idx++) {
        // y[b][c] is calculated by more than one x[b][n]
        atomicAdd(
            &y[b_idx][c_idx][d_idx], x[b_idx][n_idx][d_idx] * w
        );
    }
}

void weighted_sum(
    const torch::Tensor x,
    const torch::Tensor group,
    const torch::Tensor weights,
    torch::Tensor y) {
    const int B = x.size(0);
    const int N = x.size(1);

    const int threads = 1024;
    const int blocks = (B * N + threads - 1) / threads;

    // 根据 group 张量的类型分派不同实现
    AT_DISPATCH_INTEGRAL_TYPES(group.scalar_type(), "weighted_sum", [&] {
        weighted_sum_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            group.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            y.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &weighted_sum);
}
