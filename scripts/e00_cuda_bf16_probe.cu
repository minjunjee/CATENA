#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

__global__ void compiled_for_target_architecture(float* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        value[0] += 1.0F;
    }
}

void fail(const char* stage, const char* detail, int code) {
    std::fprintf(
        stderr,
        "{\"passed\":false,\"stage\":\"%s\",\"error\":\"%s\",\"code\":%d}\n",
        stage,
        detail,
        code
    );
    std::exit(code == 0 ? 1 : code);
}

void check_cuda(cudaError_t status, const char* stage) {
    if (status != cudaSuccess) {
        fail(stage, cudaGetErrorString(status), static_cast<int>(status));
    }
}

void check_cublas(cublasStatus_t status, const char* stage) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail(stage, "cuBLAS call failed", static_cast<int>(status));
    }
}

void check_driver(CUresult status, const char* stage) {
    if (status != CUDA_SUCCESS) {
        const char* detail = "CUDA driver API call failed";
        cuGetErrorString(status, &detail);
        fail(stage, detail, static_cast<int>(status));
    }
}

void format_uuid(const CUuuid& uuid, char* output, std::size_t output_size) {
    const unsigned char* bytes =
        reinterpret_cast<const unsigned char*>(uuid.bytes);
    const int written = std::snprintf(
        output,
        output_size,
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    );
    if (written < 0 || static_cast<std::size_t>(written) >= output_size) {
        fail("format_uuid", "UUID buffer was too small", written);
    }
}

}  // namespace

int main() {
    constexpr int n = 128;
    constexpr int elements = n * n;
    constexpr int iterations = 5;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count != 1) {
        fail("visible_device_count", "expected exactly one visible GPU", device_count);
    }

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_driver(cuInit(0), "cuInit");
    CUdevice driver_device{};
    check_driver(cuDeviceGet(&driver_device, 0), "cuDeviceGet");
    CUuuid device_uuid{};
    check_driver(cuDeviceGetUuid(&device_uuid, driver_device), "cuDeviceGetUuid");
    char uuid_text[64]{};
    format_uuid(device_uuid, uuid_text, sizeof(uuid_text));
    char pci_bus_id[32]{};
    check_cuda(
        cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0),
        "cudaDeviceGetPCIBusId"
    );

    std::vector<__nv_bfloat16> host_a(elements);
    std::vector<__nv_bfloat16> host_b(elements);
    std::vector<float> host_c(elements, 0.0F);
    for (int i = 0; i < elements; ++i) {
        host_a[i] = __float2bfloat16(1.0F);
        host_b[i] = __float2bfloat16(1.0F);
    }

    __nv_bfloat16* device_a = nullptr;
    __nv_bfloat16* device_b = nullptr;
    float* device_c = nullptr;
    check_cuda(cudaMalloc(&device_a, elements * sizeof(__nv_bfloat16)), "cudaMalloc_a");
    check_cuda(cudaMalloc(&device_b, elements * sizeof(__nv_bfloat16)), "cudaMalloc_b");
    check_cuda(cudaMalloc(&device_c, elements * sizeof(float)), "cudaMalloc_c");
    check_cuda(cudaMemset(device_c, 0, elements * sizeof(float)), "cudaMemset_c");
    compiled_for_target_architecture<<<1, 1>>>(device_c);
    check_cuda(cudaGetLastError(), "compiled_kernel_launch");
    check_cuda(cudaDeviceSynchronize(), "compiled_kernel_synchronize");
    check_cuda(
        cudaMemcpy(
            device_a,
            host_a.data(),
            elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy_a"
    );
    check_cuda(
        cudaMemcpy(
            device_b,
            host_b.data(),
            elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy_b"
    );

    cublasHandle_t handle{};
    check_cublas(cublasCreate(&handle), "cublasCreate");
    const float alpha = 1.0F;
    const float beta = 0.0F;

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate_start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate_stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord_start");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check_cublas(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                n,
                n,
                &alpha,
                device_a,
                CUDA_R_16BF,
                n,
                device_b,
                CUDA_R_16BF,
                n,
                &beta,
                device_c,
                CUDA_R_32F,
                n,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            ),
            "cublasGemmEx"
        );
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord_stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    check_cuda(
        cudaMemcpy(
            host_c.data(),
            device_c,
            elements * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy_c"
    );

    const float expected = static_cast<float>(n);
    float max_abs_error = 0.0F;
    bool finite = true;
    for (const float value : host_c) {
        finite = finite && std::isfinite(value);
        max_abs_error = std::max(max_abs_error, std::fabs(value - expected));
    }
    const bool passed = finite && max_abs_error == 0.0F;

    std::printf(
        "{\"passed\":%s,\"visible_device_count\":%d,"
        "\"name\":\"%s\",\"uuid\":\"%s\",\"pci_bus_id\":\"%s\","
        "\"compute_capability\":\"%d.%d\","
        "\"matrix_size\":%d,\"iterations\":%d,\"dtype\":\"bf16\","
        "\"accumulator\":\"fp32\",\"finite\":%s,\"max_abs_error\":%.9g,"
        "\"mean_elapsed_ms\":%.6f,\"sm_target_kernel_executed\":true}\n",
        passed ? "true" : "false",
        device_count,
        properties.name,
        uuid_text,
        pci_bus_id,
        properties.major,
        properties.minor,
        n,
        iterations,
        finite ? "true" : "false",
        max_abs_error,
        elapsed_ms / static_cast<float>(iterations)
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return passed ? 0 : 1;
}
