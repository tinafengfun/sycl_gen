#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"

#include <algorithm>
#include <stdexcept>

namespace lczero {
namespace sycldnn_backend {

#define ReportSYCLerrors(status)                                       \
  do {                                                                 \
    try {                                                              \
      status.wait_and_throw();                                         \
    } catch (const sycl::exception& e) {                               \
      throw std::runtime_error(std::string("SYCL error: ") + e.what() + " at " + \
                      __FILE__ + ":" + std::to_string(__LINE__));      \
    }                                                                  \
  } while (0)

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
inline void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

inline float warpReduce(float x, sycl::sub_group sg) {
  for (int mask = 16; mask > 0; mask >>= 1)
    x += sycl::permute_group_by_xor(sg, x, mask);
  return x;
}

inline float mishActivate(float el) {
  auto e = sycl::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  return (el <= -0.6f) ? n * d : el - 2.0f * d;
}

inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      cVal = cVal < 0 ? 0 : cVal;
      break;
    case ACTIVATION_RELU_2:
      cVal = cVal < 0 ? 0 : cVal;
      cVal *= cVal;
      break;
    case ACTIVATION_TANH:
      cVal = sycl::tanh(cVal);
      break;
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      cVal = cVal > 0 ? scale * cVal : scale * alpha * (sycl::exp(cVal) - 1.0f);
      break;
    }
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_SWISH:
      cVal /= (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_NONE:
      break;
    default:
      cVal = 0;
      break;
  }
  return cVal;
}

template <typename T>
class layer_norm_kernel;

template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, sycl::queue& stream) {
  if (C % 16 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 16384) throw std::runtime_error("unsupported filter size");

  constexpr int block_x = 32;
  int block_y = DivUp(C / 16, block_x);
  int block_z = std::min(std::max(512 / (block_x * block_y), 1), N);
  int grid_x = DivUp(N, block_z);

  sycl::range<3> grid(grid_x, 1, 1);
  sycl::range<3> block(block_z, block_y, block_x);

  auto event = stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 2> sum_acc({16, 16}, cgh);

    cgh.parallel_for<layer_norm_kernel<T>>(
        sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
          int n = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
          if (n >= N) return;
          int c = (item.get_local_id(1) * 32 + item.get_local_id(2)) * 16;
          bool oobThread = c >= C;

          int biasIndex = c;
          int tensorIndex = n * C + c;

          float val[16] = {0};
          float oth[16] = {0};

          const bool fp16 = std::is_same<sycl::half, T>::value;
          if (!oobThread) {
            if (fp16) {
              sycl::half inp[8];
              copyAs<sycl::uint4>(&inp[0], &input[tensorIndex]);
              for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &input[tensorIndex + 8]);
              for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &bias[biasIndex]);
              for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &bias[biasIndex + 8]);
              for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
              for (int i = 0; i < 16; i++) val[i] += oth[i];
            } else {
              copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
              copyAs<sycl::uint4>(&val[4], &input[tensorIndex + 4]);
              copyAs<sycl::uint4>(&val[8], &input[tensorIndex + 8]);
              copyAs<sycl::uint4>(&val[12], &input[tensorIndex + 12]);
              copyAs<sycl::uint4>(&oth[0], &bias[biasIndex]);
              copyAs<sycl::uint4>(&oth[4], &bias[biasIndex + 4]);
              copyAs<sycl::uint4>(&oth[8], &bias[biasIndex + 8]);
              copyAs<sycl::uint4>(&oth[12], &bias[biasIndex + 12]);
              for (int i = 0; i < 16; i++) val[i] += oth[i];
            }
          }

          if (!oobThread && skip != nullptr) {
            if (fp16) {
              sycl::half inp[8];
              copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex]);
              for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex + 8]);
              for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
            } else {
              copyAs<sycl::uint4>(&oth[0], &skip[tensorIndex]);
              copyAs<sycl::uint4>(&oth[4], &skip[tensorIndex + 4]);
              copyAs<sycl::uint4>(&oth[8], &skip[tensorIndex + 8]);
              copyAs<sycl::uint4>(&oth[12], &skip[tensorIndex + 12]);
            }
          }

          float s = 0;
          if (!oobThread) {
            if (skip != nullptr) {
              for (int i = 0; i < 16; i++) {
                val[i] = activate(val[i], act) * alpha + oth[i];
                s += val[i];
              }
            } else {
              for (int i = 0; i < 16; i++) {
                val[i] = activate(val[i], act) * alpha;
                s += val[i];
              }
            }
          }

          auto sg = item.get_sub_group();
          s = warpReduce(s, sg);
          if (item.get_local_id(2) == 0) {
            sum_acc[item.get_local_id(0)][item.get_local_id(1)] = s;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
            float cSum = 0;
            for (int j = 0; j < item.get_local_range(1); j++)
              cSum += sum_acc[item.get_local_id(0)][j];
            sum_acc[item.get_local_id(0)][0] = cSum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float mean = sum_acc[item.get_local_id(0)][0] / C;

          s = 0;
          if (!oobThread) {
            for (int i = 0; i < 16; i++) {
              float d = val[i] - mean;
              s += d * d;
            }
          }

          s = warpReduce(s, sg);
          if (item.get_local_id(2) == 0) {
            sum_acc[item.get_local_id(0)][item.get_local_id(1)] = s;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
            float cSum = 0;
            for (int j = 0; j < item.get_local_range(1); j++)
              cSum += sum_acc[item.get_local_id(0)][j];
            sum_acc[item.get_local_id(0)][0] = cSum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float var = sum_acc[item.get_local_id(0)][0] / C;

          if (!oobThread) {
            if (fp16) {
              sycl::half inp[8];
              copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex]);
              for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex + 8]);
              for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
            } else {
              copyAs<sycl::uint4>(&oth[0], &gammas[biasIndex]);
              copyAs<sycl::uint4>(&oth[4], &gammas[biasIndex + 4]);
              copyAs<sycl::uint4>(&oth[8], &gammas[biasIndex + 8]);
              copyAs<sycl::uint4>(&oth[12], &gammas[biasIndex + 12]);
            }
          }

          for (int i = 0; i < 16; i++) {
            float d = val[i] - mean;
            float norm = d / sycl::sqrt(var + ep);
            val[i] = norm * oth[i];
          }

          if (!oobThread) {
            if (fp16) {
              sycl::half inp[8];
              copyAs<sycl::uint4>(&inp[0], &betas[biasIndex]);
              for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &betas[biasIndex + 8]);
              for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
            } else {
              copyAs<sycl::uint4>(&oth[0], &betas[biasIndex]);
              copyAs<sycl::uint4>(&oth[4], &betas[biasIndex + 4]);
              copyAs<sycl::uint4>(&oth[8], &betas[biasIndex + 8]);
              copyAs<sycl::uint4>(&oth[12], &betas[biasIndex + 12]);
            }
          }

          for (int i = 0; i < 16; i++) {
            val[i] += oth[i];
          }

          if (!oobThread) {
            if (fp16) {
              sycl::half op[8];
              for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i];
              copyAs<sycl::uint4>(&output[tensorIndex], &op[0]);
              for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i + 8];
              copyAs<sycl::uint4>(&output[tensorIndex + 8], &op[0]);
            } else {
              copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
              copyAs<sycl::uint4>(&output[tensorIndex + 4], &val[4]);
              copyAs<sycl::uint4>(&output[tensorIndex + 8], &val[8]);
              copyAs<sycl::uint4>(&output[tensorIndex + 12], &val[12]);
            }
          }
        });
  });
  ReportSYCLerrors(event);
}

template void LayerNorm<float>(int N, int C, float* output, const float* input,
                               const float* bias, const float* skip,
                               const float* gammas, const float* betas,
                               float ep, float alpha, ActivationFunction act,
                               sycl::queue& stream);

template void LayerNorm<sycl::half>(int N, int C, sycl::half* output,
                                    const sycl::half* input,
                                    const sycl::half* bias,
                                    const sycl::half* skip,
                                    const sycl::half* gammas,
                                    const sycl::half* betas, float ep,
                                    float alpha, ActivationFunction act,
                                    sycl::queue& stream);

}  // namespace cudnn_backend
}  // namespace lczero