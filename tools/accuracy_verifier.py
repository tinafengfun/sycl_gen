#!/usr/bin/env python3
"""
Accuracy Verifier Component
准确度验证组件 - 独立、可复用的CUDA vs SYCL数值验证模块

设计原则:
1. 单一职责: 只做准确度验证，不涉及转换逻辑
2. 可配置: 支持不同的验证策略和容忍度
3. 可扩展: 易于添加新的kernel类型支持
4. 可测试: 提供mock和测试接口

使用示例:
    verifier = AccuracyVerifier(
        cuda_platform=CUDARemotePlatform("10.112.229.160"),
        sycl_platform=SYCLLocalPlatform("lsv-container"),
        tolerance=AdaptiveTolerance()
    )
    
    result = await verifier.verify(kernel_id="softmax")
    if result.passed:
        print(f"✅ MAE={result.mae:.2e}")
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

# 添加 cuda-sycl-converter harness 路径
import sys
CUDA_SYCL_CONVERTER_PATH = Path(__file__).parent.parent / 'cuda-sycl-converter' / 'src'
if str(CUDA_SYCL_CONVERTER_PATH) not in sys.path:
    sys.path.insert(0, str(CUDA_SYCL_CONVERTER_PATH))

# 尝试导入 cuda-sycl-converter 的 harnesses
try:
    from harnesses.all_harnesses import ALL_HARNESSES
    from harnesses.batch4_harnesses import PHASE5_BATCH4_HARNESSES
    CUDA_SYCL_HARNESSES_AVAILABLE = True
    logger_import = logging.getLogger(__name__)
    logger_import.info(f"✅ Loaded {len(ALL_HARNESSES)} kernels from cuda-sycl-converter")
except ImportError as e:
    CUDA_SYCL_HARNESSES_AVAILABLE = False
    ALL_HARNESSES = {}
    PHASE5_BATCH4_HARNESSES = {}
    logger_import = logging.getLogger(__name__)
    logger_import.warning(f"⚠️ Could not import cuda-sycl-converter harnesses: {e}")


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """验证状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """准确度验证结果"""
    kernel_id: str
    status: VerificationStatus
    mae: float = 0.0
    max_error: float = 0.0
    pass_rate: float = 0.0
    tolerance_used: Dict = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def passed(self) -> bool:
        """是否通过验证"""
        return self.status == VerificationStatus.PASSED
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'kernel_id': self.kernel_id,
            'status': self.status.value,
            'mae': self.mae,
            'max_error': self.max_error,
            'pass_rate': self.pass_rate,
            'tolerance_used': self.tolerance_used,
            'error_message': self.error_message,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp
        }


@dataclass
class ToleranceConfig:
    """容忍度配置"""
    abs_tolerance: float = 1e-5
    rel_tolerance: float = 1e-4
    pass_rate_threshold: float = 0.95
    
    # 针对不同kernel类型的特殊配置
    kernel_specific: Dict[str, Dict] = field(default_factory=lambda: {
        'fp16': {'abs': 1e-3, 'rel': 1e-2},
        'softmax': {'abs': 1e-3, 'rel': 5e-3},
        'winograd': {'abs': 1e-2, 'rel': 1e-1},
    })
    
    def get_for_kernel(self, kernel_type: str) -> Dict[str, float]:
        """获取特定kernel类型的容忍度"""
        if kernel_type in self.kernel_specific:
            cfg = self.kernel_specific[kernel_type]
            return {
                'abs': cfg['abs'],
                'rel': cfg['rel'],
                'pass_rate': self.pass_rate_threshold
            }
        return {
            'abs': self.abs_tolerance,
            'rel': self.rel_tolerance,
            'pass_rate': self.pass_rate_threshold
        }


class ExecutionPlatform(ABC):
    """执行平台抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def compile_and_run(
        self, 
        source_code: str, 
        output_file: str,
        timeout: int = 120
    ) -> Tuple[bool, Optional[str]]:
        """
        编译并运行代码
        
        Args:
            source_code: 源代码
            output_file: 输出文件路径
            timeout: 超时时间
            
        Returns:
            (成功标志, 错误信息)
        """
        pass
    
    @abstractmethod
    async def get_output(self, remote_path: str, local_path: str) -> bool:
        """获取输出文件"""
        pass


class CUDARemotePlatform(ExecutionPlatform):
    """CUDA远程平台"""
    
    def __init__(self, host: str, container: str = "cuda12.9-test"):
        super().__init__("cuda_remote")
        self.host = host
        self.container = container
    
    async def compile_and_run(
        self, 
        source_code: str, 
        output_file: str,
        timeout: int = 120
    ) -> Tuple[bool, Optional[str]]:
        """在远程CUDA环境编译运行"""
        try:
            # 保存源代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(source_code)
                local_file = f.name
            
            # 复制到远程
            scp_cmd = ['scp', '-o', 'StrictHostKeyChecking=no', 
                      local_file, f'{self.host}:/tmp/test.cu']
            subprocess.run(scp_cmd, capture_output=True, timeout=30, check=True)
            
            # 编译运行
            cmd = f'''
            ssh -o StrictHostKeyChecking=no {self.host} "
                docker cp /tmp/test.cu {self.container}:/workspace/test.cu &&
                docker exec {self.container} bash -c '
                    cd /workspace && 
                    nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test 2>&1 &&
                    ./test 2>&1
                '
            "
            '''
            
            result = subprocess.run(
                cmd, shell=True, capture_output=True, 
                text=True, timeout=timeout
            )
            
            # 清理
            Path(local_file).unlink(missing_ok=True)
            
            if result.returncode != 0:
                return False, result.stderr or result.stdout
            
            return True, None
            
        except subprocess.TimeoutExpired:
            return False, "Compilation/execution timeout"
        except Exception as e:
            return False, str(e)
    
    async def get_output(self, remote_path: str, local_path: str) -> bool:
        """从远程获取输出文件"""
        try:
            # 从容器复制到主机
            cmd1 = f"ssh -o StrictHostKeyChecking=no {self.host} \"docker cp {self.container}:{remote_path} /tmp/\""
            subprocess.run(cmd1, shell=True, capture_output=True, timeout=10, check=True)
            
            # 从主机复制到本地
            cmd2 = ['scp', '-o', 'StrictHostKeyChecking=no',
                   f'{self.host}:/tmp/{Path(remote_path).name}', local_path]
            subprocess.run(cmd2, capture_output=True, timeout=10, check=True)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to get output: {e}")
            return False


class SYCLLocalPlatform(ExecutionPlatform):
    """SYCL本地Docker平台"""
    
    def __init__(self, container: str = "lsv-container"):
        super().__init__("sycl_local")
        self.container = container
    
    async def compile_and_run(
        self, 
        source_code: str, 
        output_file: str,
        timeout: int = 120
    ) -> Tuple[bool, Optional[str]]:
        """在本地SYCL容器编译运行"""
        try:
            # 保存源代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(source_code)
                local_file = f.name
            
            # 复制到容器
            cp_cmd = ['docker', 'cp', local_file, f'{self.container}:/workspace/test.cpp']
            subprocess.run(cp_cmd, capture_output=True, timeout=10, check=True)
            
            # 编译运行
            cmd = [
                'docker', 'exec', self.container, 'bash', '-c',
                'cd /workspace && icpx -fsycl -O2 test.cpp -o test 2>&1 && ./test 2>&1'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            # 清理
            Path(local_file).unlink(missing_ok=True)
            
            if result.returncode != 0:
                return False, result.stderr or result.stdout
            
            return True, None
            
        except subprocess.TimeoutExpired:
            return False, "Compilation/execution timeout"
        except Exception as e:
            return False, str(e)
    
    async def get_output(self, remote_path: str, local_path: str) -> bool:
        """从容器获取输出文件"""
        try:
            cmd = ['docker', 'cp', f'{self.container}:{remote_path}', local_path]
            subprocess.run(cmd, capture_output=True, timeout=10, check=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to get output: {e}")
            return False


class HarnessGenerator:
    """Test Harness生成器"""
    
    # 内置harness模板
    TEMPLATES = {
        'copy_type_converted': {
            'cuda': '''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(half* o, const float* i, int n) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n) o[idx]=__float2half(i[idx]);
}
int main(){
    const int N=1024;
    float h_i[N]; half h_o[N];
    for(int i=0;i<N;i++)h_i[i]=sinf(i*0.01f)*0.9f;
    float* d_i; half* d_o;
    cudaMalloc(&d_i,N*sizeof(float));cudaMalloc(&d_o,N*sizeof(half));
    cudaMemcpy(d_i,h_i,N*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(N+255)/256,256>>>(d_o,d_i,N);
    cudaMemcpy(h_o,d_o,N*sizeof(half),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(half),N,f);fclose(f);
    cudaFree(d_i);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=1024;
    float h_i[N]; sycl::half h_o[N];
    for(int i=0;i<N;i++)h_i[i]=sycl::sin(i*0.01f)*0.9f;
    float* d_i=sycl::malloc_device<float>(N,q);
    sycl::half* d_o=sycl::malloc_device<sycl::half>(N,q);
    q.memcpy(d_i,h_i,N*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(N),[=](sycl::id<1> idx){
        d_o[idx]=sycl::half(d_i[idx]);
    }).wait();
    q.memcpy(h_o,d_o,N*sizeof(sycl::half)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),N*sizeof(sycl::half));
    sycl::free(d_i,q);sycl::free(d_o,q);return 0;
}'''
        },
        'global_avg_pool': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,int N,int C,int H,int W){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=N*C;
    if(idx<total){
        int n=idx/C,c=idx%C;
        double s=0;
        for(int h=0;h<H;h++)for(int w=0;w<W;w++)
            s+=i[((n*C+c)*H+h)*W+w];
        o[idx]=s/(H*W);
    }
}
int main(){
    const int N=2,C=32,H=8,W=8;
    const int in_sz=N*C*H*W,out_sz=N*C;
    float h_i[in_sz],h_o[out_sz];
    for(int i=0;i<in_sz;i++)h_i[i]=sinf(i*0.01f)*0.5f+0.5f;
    float *d_i,*d_o;
    cudaMalloc(&d_i,in_sz*sizeof(float));cudaMalloc(&d_o,out_sz*sizeof(float));
    cudaMemcpy(d_i,h_i,in_sz*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(out_sz+255)/256,256>>>(d_o,d_i,N,C,H,W);
    cudaMemcpy(h_o,d_o,out_sz*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),out_sz,f);fclose(f);
    cudaFree(d_i);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=2,C=32,H=8,W=8;
    const int in_sz=N*C*H*W,out_sz=N*C;
    float h_i[in_sz],h_o[out_sz];
    for(int i=0;i<in_sz;i++)h_i[i]=sycl::sin(i*0.01f)*0.5f+0.5f;
    float* d_i=sycl::malloc_device<float>(in_sz,q);
    float* d_o=sycl::malloc_device<float>(out_sz,q);
    q.memcpy(d_i,h_i,in_sz*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(out_sz),[=](sycl::id<1> idx){
        int n=idx[0]/C,c=idx[0]%C;
        double s=0;
        for(int h=0;h<H;h++)for(int w=0;w<W;w++)
            s+=d_i[((n*C+c)*H+h)*W+w];
        d_o[idx[0]]=s/(H*W);
    }).wait();
    q.memcpy(h_o,d_o,out_sz*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),out_sz*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_o,q);return 0;
}'''
        },
        'softmax': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,int N,int C){
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n<N){
        float mx=i[n*C];
        for(int c=1;c<C;c++)mx=fmaxf(mx,i[n*C+c]);
        float s=0;
        for(int c=0;c<C;c++){float e=expf(i[n*C+c]-mx);o[n*C+c]=e;s+=e;}
        for(int c=0;c<C;c++)o[n*C+c]/=s;
    }
}
int main(){
    const int N=4,C=128,sz=N*C;
    float h_i[sz],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sinf(i*0.01f)*2.0f;
    float *d_i,*d_o;
    cudaMalloc(&d_i,sz*sizeof(float));cudaMalloc(&d_o,sz*sizeof(float));
    cudaMemcpy(d_i,h_i,sz*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(N+255)/256,256>>>(d_o,d_i,N,C);
    cudaMemcpy(h_o,d_o,sz*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),sz,f);fclose(f);
    cudaFree(d_i);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=4,C=128,sz=N*C;
    float h_i[sz],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sycl::sin(i*0.01f)*2.0f;
    float* d_i=sycl::malloc_device<float>(sz,q);
    float* d_o=sycl::malloc_device<float>(sz,q);
    q.memcpy(d_i,h_i,sz*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(N),[=](sycl::id<1> idx){
        int n=idx[0];
        float mx=d_i[n*C];
        for(int c=1;c<C;c++)mx=sycl::fmax(mx,d_i[n*C+c]);
        float s=0;
        for(int c=0;c<C;c++){float e=sycl::exp(d_i[n*C+c]-mx);d_o[n*C+c]=e;s+=e;}
        for(int c=0;c<C;c++)d_o[n*C+c]/=s;
    }).wait();
    q.memcpy(h_o,d_o,sz*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),sz*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_o,q);return 0;
}'''
        },
        'softmax_opt_64': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,int N){
    const int C=64;
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n<N){
        float mx=i[n*C];
        #pragma unroll
        for(int c=1;c<C;c++)mx=fmaxf(mx,i[n*C+c]);
        float s=0;
        #pragma unroll
        for(int c=0;c<C;c++){float e=expf(i[n*C+c]-mx);o[n*C+c]=e;s+=e;}
        #pragma unroll
        for(int c=0;c<C;c++)o[n*C+c]/=s;
    }
}
int main(){
    const int N=8,C=64,sz=N*C;
    float h_i[sz],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sinf(i*0.01f)*2.0f;
    float *d_i,*d_o;
    cudaMalloc(&d_i,sz*sizeof(float));cudaMalloc(&d_o,sz*sizeof(float));
    cudaMemcpy(d_i,h_i,sz*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(N+255)/256,256>>>(d_o,d_i,N);
    cudaMemcpy(h_o,d_o,sz*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),sz,f);fclose(f);
    cudaFree(d_i);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=8,C=64,sz=N*C;
    float h_i[sz],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sycl::sin(i*0.01f)*2.0f;
    float* d_i=sycl::malloc_device<float>(sz,q);
    float* d_o=sycl::malloc_device<float>(sz,q);
    q.memcpy(d_i,h_i,sz*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(N),[=](sycl::id<1> idx){
        int n=idx[0];
        const int C=64;
        float mx=d_i[n*C];
        #pragma unroll
        for(int c=1;c<C;c++)mx=sycl::fmax(mx,d_i[n*C+c]);
        float s=0;
        #pragma unroll
        for(int c=0;c<C;c++){float e=sycl::exp(d_i[n*C+c]-mx);d_o[n*C+c]=e;s+=e;}
        #pragma unroll
        for(int c=0;c<C;c++)d_o[n*C+c]/=s;
    }).wait();
    q.memcpy(h_o,d_o,sz*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),sz*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_o,q);return 0;
}'''
        },
        'winograd_input_transform': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,int N,int C,int H,int W){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int sz=N*C*H*W;
    if(idx<sz)o[idx]=i[idx]*0.25f;
}
int main(){
    const int N=2,C=32,H=8,W=8,sz=N*C*H*W;
    float h_i[sz],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sinf(i*0.01f)*0.5f;
    float *d_i,*d_o;
    cudaMalloc(&d_i,sz*sizeof(float));cudaMalloc(&d_o,sz*sizeof(float));
    cudaMemcpy(d_i,h_i,sz*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(sz+255)/256,256>>>(d_o,d_i,N,C,H,W);
    cudaMemcpy(h_o,d_o,sz*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),sz,f);fclose(f);
    cudaFree(d_i);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=2,C=32,H=8,W=8,sz=N*C*H*W;
    float h_i[sz],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sycl::sin(i*0.01f)*0.5f;
    float* d_i=sycl::malloc_device<float>(sz,q);
    float* d_o=sycl::malloc_device<float>(sz,q);
    q.memcpy(d_i,h_i,sz*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(sz),[=](sycl::id<1> idx){
        d_o[idx]=d_i[idx]*0.25f;
    }).wait();
    q.memcpy(h_o,d_o,sz*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),sz*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_o,q);return 0;
}'''
        },
        'add_vectors': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* a,const float* b,int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N)o[idx]=a[idx]+b[idx];
}
int main(){
    const int N=1024;
    float h_a[N],h_b[N],h_o[N];
    for(int i=0;i<N;i++){h_a[i]=sinf(i*0.01f)*0.5f;h_b[i]=cosf(i*0.02f)*0.3f;}
    float *d_a,*d_b,*d_o;
    cudaMalloc(&d_a,N*sizeof(float));cudaMalloc(&d_b,N*sizeof(float));cudaMalloc(&d_o,N*sizeof(float));
    cudaMemcpy(d_a,h_a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(N+255)/256,256>>>(d_o,d_a,d_b,N);
    cudaMemcpy(h_o,d_o,N*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),N,f);fclose(f);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=1024;
    float h_a[N],h_b[N],h_o[N];
    for(int i=0;i<N;i++){h_a[i]=sycl::sin(i*0.01f)*0.5f;h_b[i]=sycl::cos(i*0.02f)*0.3f;}
    float* d_a=sycl::malloc_device<float>(N,q);
    float* d_b=sycl::malloc_device<float>(N,q);
    float* d_o=sycl::malloc_device<float>(N,q);
    q.memcpy(d_a,h_a,N*sizeof(float)).wait();
    q.memcpy(d_b,h_b,N*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(N),[=](sycl::id<1> idx){
        d_o[idx[0]]=d_a[idx[0]]+d_b[idx[0]];
    }).wait();
    q.memcpy(h_o,d_o,N*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),N*sizeof(float));
    sycl::free(d_a,q);sycl::free(d_b,q);sycl::free(d_o,q);return 0;
}'''
        },
        'add_bias_batched': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,const float* b,int N,int C){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N*C){
        int c=idx%C;
        o[idx]=i[idx]+b[c];
    }
}
int main(){
    const int N=4,C=64,sz=N*C;
    float h_i[sz],h_b[C],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sinf(i*0.01f)*0.5f;
    for(int c=0;c<C;c++)h_b[c]=cosf(c*0.02f)*0.3f;
    float *d_i,*d_b,*d_o;
    cudaMalloc(&d_i,sz*sizeof(float));cudaMalloc(&d_b,C*sizeof(float));cudaMalloc(&d_o,sz*sizeof(float));
    cudaMemcpy(d_i,h_i,sz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,C*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(sz+255)/256,256>>>(d_o,d_i,d_b,N,C);
    cudaMemcpy(h_o,d_o,sz*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),sz,f);fclose(f);
    cudaFree(d_i);cudaFree(d_b);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=4,C=64,sz=N*C;
    float h_i[sz],h_b[C],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sycl::sin(i*0.01f)*0.5f;
    for(int c=0;c<C;c++)h_b[c]=sycl::cos(c*0.02f)*0.3f;
    float* d_i=sycl::malloc_device<float>(sz,q);
    float* d_b=sycl::malloc_device<float>(C,q);
    float* d_o=sycl::malloc_device<float>(sz,q);
    q.memcpy(d_i,h_i,sz*sizeof(float)).wait();
    q.memcpy(d_b,h_b,C*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(sz),[=](sycl::id<1> idx){
        int c=idx[0]%C;
        d_o[idx[0]]=d_i[idx[0]]+d_b[c];
    }).wait();
    q.memcpy(h_o,d_o,sz*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),sz*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_b,q);sycl::free(d_o,q);return 0;
}'''
        },
        'global_scale': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,const float* s,int N,int C){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N*C){
        int c=idx%C;
        o[idx]=i[idx]*s[c];
    }
}
int main(){
    const int N=4,C=64,sz=N*C;
    float h_i[sz],h_s[C],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sinf(i*0.01f)*0.5f;
    for(int c=0;c<C;c++)h_s[c]=0.9f+c*0.01f;
    float *d_i,*d_s,*d_o;
    cudaMalloc(&d_i,sz*sizeof(float));cudaMalloc(&d_s,C*sizeof(float));cudaMalloc(&d_o,sz*sizeof(float));
    cudaMemcpy(d_i,h_i,sz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_s,h_s,C*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(sz+255)/256,256>>>(d_o,d_i,d_s,N,C);
    cudaMemcpy(h_o,d_o,sz*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),sz,f);fclose(f);
    cudaFree(d_i);cudaFree(d_s);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=4,C=64,sz=N*C;
    float h_i[sz],h_s[C],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sycl::sin(i*0.01f)*0.5f;
    for(int c=0;c<C;c++)h_s[c]=0.9f+c*0.01f;
    float* d_i=sycl::malloc_device<float>(sz,q);
    float* d_s=sycl::malloc_device<float>(C,q);
    float* d_o=sycl::malloc_device<float>(sz,q);
    q.memcpy(d_i,h_i,sz*sizeof(float)).wait();
    q.memcpy(d_s,h_s,C*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(sz),[=](sycl::id<1> idx){
        int c=idx[0]%C;
        d_o[idx[0]]=d_i[idx[0]]*d_s[c];
    }).wait();
    q.memcpy(h_o,d_o,sz*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),sz*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_s,q);sycl::free(d_o,q);return 0;
}'''
        },
        'batch_norm': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,const float* m,const float* v,int N,int C){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N*C){
        int c=idx%C;
        o[idx]=(i[idx]-m[c])*rsqrtf(v[c]+1e-5f);
    }
}
int main(){
    const int N=4,C=64,sz=N*C;
    float h_i[sz],h_m[C],h_v[C],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sinf(i*0.01f)*0.5f;
    for(int c=0;c<C;c++){h_m[c]=0.1f*c;h_v[c]=0.01f;}
    float *d_i,*d_m,*d_v,*d_o;
    cudaMalloc(&d_i,sz*sizeof(float));cudaMalloc(&d_m,C*sizeof(float));
    cudaMalloc(&d_v,C*sizeof(float));cudaMalloc(&d_o,sz*sizeof(float));
    cudaMemcpy(d_i,h_i,sz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,h_m,C*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,h_v,C*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(sz+255)/256,256>>>(d_o,d_i,d_m,d_v,N,C);
    cudaMemcpy(h_o,d_o,sz*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),sz,f);fclose(f);
    cudaFree(d_i);cudaFree(d_m);cudaFree(d_v);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=4,C=64,sz=N*C;
    float h_i[sz],h_m[C],h_v[C],h_o[sz];
    for(int i=0;i<sz;i++)h_i[i]=sycl::sin(i*0.01f)*0.5f;
    for(int c=0;c<C;c++){h_m[c]=0.1f*c;h_v[c]=0.01f;}
    float* d_i=sycl::malloc_device<float>(sz,q);
    float* d_m=sycl::malloc_device<float>(C,q);
    float* d_v=sycl::malloc_device<float>(C,q);
    float* d_o=sycl::malloc_device<float>(sz,q);
    q.memcpy(d_i,h_i,sz*sizeof(float)).wait();
    q.memcpy(d_m,h_m,C*sizeof(float)).wait();
    q.memcpy(d_v,h_v,C*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(sz),[=](sycl::id<1> idx){
        int c=idx[0]%C;
        d_o[idx[0]]=(d_i[idx[0]]-d_m[c])*sycl::rsqrt(d_v[c]+1e-5f);
    }).wait();
    q.memcpy(h_o,d_o,sz*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),sz*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_m,q);sycl::free(d_v,q);sycl::free(d_o,q);return 0;
}'''
        },
        'policy_map': {
            'cuda': '''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void kernel(float* o,const float* i,int N,int C){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        int max_c=0;
        float max_v=i[idx*C];
        for(int c=1;c<C;c++){
            if(i[idx*C+c]>max_v){max_v=i[idx*C+c];max_c=c;}
        }
        o[idx]=max_c;
    }
}
int main(){
    const int N=4,C=128,sz=N*C;
    float h_i[sz],h_o[N];
    for(int i=0;i<sz;i++)h_i[i]=sinf(i*0.01f)*2.0f;
    float *d_i,*d_o;
    cudaMalloc(&d_i,sz*sizeof(float));cudaMalloc(&d_o,N*sizeof(float));
    cudaMemcpy(d_i,h_i,sz*sizeof(float),cudaMemcpyHostToDevice);
    kernel<<<(N+255)/256,256>>>(d_o,d_i,N,C);
    cudaMemcpy(h_o,d_o,N*sizeof(float),cudaMemcpyDeviceToHost);
    FILE*f=fopen("/workspace/output_cuda.bin","wb");
    fwrite(h_o,sizeof(float),N,f);fclose(f);
    cudaFree(d_i);cudaFree(d_o);return 0;
}''',
            'sycl': '''#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main(){
    sycl::queue q(sycl::gpu_selector_v);
    const int N=4,C=128,sz=N*C;
    float h_i[sz],h_o[N];
    for(int i=0;i<sz;i++)h_i[i]=sycl::sin(i*0.01f)*2.0f;
    float* d_i=sycl::malloc_device<float>(sz,q);
    float* d_o=sycl::malloc_device<float>(N,q);
    q.memcpy(d_i,h_i,sz*sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(N),[=](sycl::id<1> idx){
        int max_c=0;
        float max_v=d_i[idx[0]*C];
        for(int c=1;c<C;c++){
            if(d_i[idx[0]*C+c]>max_v){max_v=d_i[idx[0]*C+c];max_c=c;}
        }
        d_o[idx[0]]=max_c;
    }).wait();
    q.memcpy(h_o,d_o,N*sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin",std::ios::binary);
    f.write(reinterpret_cast<char*>(h_o),N*sizeof(float));
    sycl::free(d_i,q);sycl::free(d_o,q);return 0;
}'''
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 合并所有可用的 harnesses
        self._all_harnesses = {}
        self._all_harnesses.update(self.TEMPLATES)
        
        # 优先使用 cuda-sycl-converter 的 harnesses
        if CUDA_SYCL_HARNESSES_AVAILABLE:
            # ALL_HARNESSES 和 PHASE5_BATCH4_HARNESSES 格式为 {'kernel_id': {'cuda': '...', 'sycl': '...'}}
            for kernel_id, harness_data in ALL_HARNESSES.items():
                if isinstance(harness_data, dict) and 'cuda' in harness_data and 'sycl' in harness_data:
                    self._all_harnesses[kernel_id] = harness_data
                    
            for kernel_id, harness_data in PHASE5_BATCH4_HARNESSES.items():
                if isinstance(harness_data, dict) and 'cuda' in harness_data and 'sycl' in harness_data:
                    self._all_harnesses[kernel_id] = harness_data
            
            self.logger.info(f"✅ HarnessGenerator initialized with {len(self._all_harnesses)} kernels")
        else:
            self.logger.warning("⚠️ Using only built-in harness templates")
    
    def generate(self, kernel_id: str, platform: str) -> Optional[str]:
        """
        生成harness代码 - 优先使用 cuda-sycl-converter 的 harnesses
        
        Args:
            kernel_id: 内核标识符
            platform: 'cuda' 或 'sycl'
            
        Returns:
            harness代码或None
        """
        # 首先检查合并后的 harnesses（包含 cuda-sycl-converter 的）
        if kernel_id in self._all_harnesses:
            code = self._all_harnesses[kernel_id].get(platform)
            if code:
                # 记录使用的 harness 来源
                if kernel_id in ALL_HARNESSES or kernel_id in PHASE5_BATCH4_HARNESSES:
                    self.logger.debug(f"Using cuda-sycl-converter harness for {kernel_id}/{platform}")
                return code
        
        # 最后回退到内置模板
        if kernel_id in self.TEMPLATES:
            code = self.TEMPLATES[kernel_id].get(platform)
            if code:
                self.logger.debug(f"Using built-in template for {kernel_id}/{platform}")
                return code
        
        self.logger.warning(f"No harness template for {kernel_id} on {platform}")
        return None
    
    def list_available_kernels(self) -> List[str]:
        """列出所有可用的 kernel IDs"""
        return list(self._all_harnesses.keys())
    
    def get_stats(self) -> Dict[str, int]:
        """获取 harness 统计信息"""
        return {
            'total': len(self._all_harnesses),
            'builtin': len(self.TEMPLATES),
            'from_cuda_sycl_converter': len(ALL_HARNESSES) + len(PHASE5_BATCH4_HARNESSES)
        }
    
    def register_template(self, kernel_id: str, cuda_code: str, sycl_code: str):
        """注册新的harness模板"""
        self.TEMPLATES[kernel_id] = {'cuda': cuda_code, 'sycl': sycl_code}


class AccuracyVerifier:
    """
    准确度验证器 - 核心组件
    
    功能:
    1. 生成并执行测试harness
    2. 比较CUDA和SYCL输出
    3. 计算误差指标
    4. 生成验证报告
    """
    
    def __init__(
        self,
        cuda_platform: Optional[ExecutionPlatform] = None,
        sycl_platform: Optional[ExecutionPlatform] = None,
        harness_generator: Optional[HarnessGenerator] = None,
        tolerance: Optional[ToleranceConfig] = None
    ):
        """
        初始化验证器
        
        Args:
            cuda_platform: CUDA执行平台
            sycl_platform: SYCL执行平台
            harness_generator: Harness生成器
            tolerance: 容忍度配置
        """
        self.cuda_platform = cuda_platform or CUDARemotePlatform("10.112.229.160")
        self.sycl_platform = sycl_platform or SYCLLocalPlatform("lsv-container")
        self.harness_gen = harness_generator or HarnessGenerator()
        self.tolerance = tolerance or ToleranceConfig()
        self.logger = logging.getLogger(__name__)
        
        # 结果缓存
        self._cache: Dict[str, VerificationResult] = {}
    
    async def verify(
        self, 
        kernel_id: str,
        use_cache: bool = True,
        kernel_type: Optional[str] = None
    ) -> VerificationResult:
        """
        验证指定kernel的准确度
        
        Args:
            kernel_id: 内核标识符
            use_cache: 是否使用缓存
            kernel_type: 内核类型(用于选择容忍度)
            
        Returns:
            VerificationResult对象
        """
        import time
        start_time = time.time()
        
        # 检查缓存
        if use_cache and kernel_id in self._cache:
            self.logger.info(f"Using cached result for {kernel_id}")
            return self._cache[kernel_id]
        
        self.logger.info(f"Starting verification for {kernel_id}")
        
        # 获取harness代码
        cuda_harness = self.harness_gen.generate(kernel_id, 'cuda')
        sycl_harness = self.harness_gen.generate(kernel_id, 'sycl')
        
        if not cuda_harness or not sycl_harness:
            result = VerificationResult(
                kernel_id=kernel_id,
                status=VerificationStatus.ERROR,
                error_message=f"No harness template for {kernel_id}",
                duration_seconds=time.time() - start_time
            )
            return result
        
        # 并行执行CUDA和SYCL测试
        cuda_task = self.cuda_platform.compile_and_run(
            cuda_harness, "/workspace/output_cuda.bin"
        )
        sycl_task = self.sycl_platform.compile_and_run(
            sycl_harness, "/workspace/output_sycl.bin"
        )
        
        cuda_success, cuda_error = await cuda_task
        sycl_success, sycl_error = await sycl_task
        
        # 检查执行结果
        if not cuda_success:
            result = VerificationResult(
                kernel_id=kernel_id,
                status=VerificationStatus.ERROR,
                error_message=f"CUDA execution failed: {cuda_error}",
                duration_seconds=time.time() - start_time
            )
            return result
        
        if not sycl_success:
            result = VerificationResult(
                kernel_id=kernel_id,
                status=VerificationStatus.ERROR,
                error_message=f"SYCL execution failed: {sycl_error}",
                duration_seconds=time.time() - start_time
            )
            return result
        
        # 获取输出文件
        cuda_local = "/tmp/cuda_output.bin"
        sycl_local = "/tmp/sycl_output.bin"
        
        cuda_get = await self.cuda_platform.get_output("/workspace/output_cuda.bin", cuda_local)
        sycl_get = await self.sycl_platform.get_output("/workspace/output_sycl.bin", sycl_local)
        
        if not cuda_get or not sycl_get:
            result = VerificationResult(
                kernel_id=kernel_id,
                status=VerificationStatus.ERROR,
                error_message="Failed to retrieve output files",
                duration_seconds=time.time() - start_time
            )
            return result
        
        # 比较输出
        try:
            cuda_output = np.fromfile(cuda_local, dtype=np.float32)
            sycl_output = np.fromfile(sycl_local, dtype=np.float32)
            
            if len(cuda_output) != len(sycl_output):
                result = VerificationResult(
                    kernel_id=kernel_id,
                    status=VerificationStatus.ERROR,
                    error_message=f"Output size mismatch: CUDA={len(cuda_output)}, SYCL={len(sycl_output)}",
                    duration_seconds=time.time() - start_time
                )
                return result
            
            # 计算误差
            diff = np.abs(cuda_output - sycl_output)
            mae = float(np.mean(diff))
            max_error = float(np.max(diff))
            
            # 计算通过率
            kernel_type = kernel_type or self._infer_kernel_type(kernel_id)
            tolerance = self.tolerance.get_for_kernel(kernel_type)
            
            passed_count = 0
            for i in range(len(cuda_output)):
                abs_ok = diff[i] < tolerance['abs']
                rel_ok = diff[i] / max(abs(cuda_output[i]), 1e-10) < tolerance['rel']
                if abs_ok or rel_ok:
                    passed_count += 1
            
            pass_rate = passed_count / len(cuda_output)
            passed = pass_rate >= tolerance['pass_rate']
            
            result = VerificationResult(
                kernel_id=kernel_id,
                status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                mae=mae,
                max_error=max_error,
                pass_rate=pass_rate,
                tolerance_used=tolerance,
                duration_seconds=time.time() - start_time
            )
            
            # 缓存结果
            if use_cache:
                self._cache[kernel_id] = result
            
            return result
            
        except Exception as e:
            result = VerificationResult(
                kernel_id=kernel_id,
                status=VerificationStatus.ERROR,
                error_message=f"Comparison failed: {str(e)}",
                duration_seconds=time.time() - start_time
            )
            return result
    
    def _infer_kernel_type(self, kernel_id: str) -> str:
        """从kernel_id推断类型"""
        if 'fp16' in kernel_id or 'half' in kernel_id:
            return 'fp16'
        elif 'softmax' in kernel_id:
            return 'softmax'
        elif 'winograd' in kernel_id:
            return 'winograd'
        return 'default'
    
    async def verify_batch(
        self, 
        kernel_ids: List[str],
        max_concurrency: int = 3,
        progress_callback: Optional[Callable[[str, VerificationResult], None]] = None
    ) -> Dict[str, VerificationResult]:
        """
        批量验证多个kernel
        
        Args:
            kernel_ids: 内核ID列表
            max_concurrency: 最大并发数
            progress_callback: 进度回调函数
            
        Returns:
            结果字典 {kernel_id: result}
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def verify_with_limit(kernel_id: str):
            async with semaphore:
                result = await self.verify(kernel_id)
                results[kernel_id] = result
                if progress_callback:
                    progress_callback(kernel_id, result)
                return result
        
        tasks = [verify_with_limit(kid) for kid in kernel_ids]
        await asyncio.gather(*tasks)
        
        return results
    
    def clear_cache(self):
        """清除结果缓存"""
        self._cache.clear()
    
    def get_cached_result(self, kernel_id: str) -> Optional[VerificationResult]:
        """获取缓存的结果"""
        return self._cache.get(kernel_id)


# 便捷函数
async def verify_kernel_accuracy(
    kernel_id: str,
    cuda_host: str = "10.112.229.160",
    sycl_container: str = "lsv-container"
) -> VerificationResult:
    """
    便捷函数: 验证单个kernel的准确度
    
    Args:
        kernel_id: 内核标识符
        cuda_host: CUDA远程主机地址
        sycl_container: SYCL容器名称
        
    Returns:
        VerificationResult
        
    Example:
        >>> result = await verify_kernel_accuracy("softmax")
        >>> print(f"Passed: {result.passed}, MAE: {result.mae}")
    """
    verifier = AccuracyVerifier(
        cuda_platform=CUDARemotePlatform(cuda_host),
        sycl_platform=SYCLLocalPlatform(sycl_container)
    )
    return await verifier.verify(kernel_id)


if __name__ == "__main__":
    # 简单测试
    async def test():
        print("🧪 Testing AccuracyVerifier...")
        
        # 创建验证器
        verifier = AccuracyVerifier()
        
        # 测试单个kernel
        result = await verifier.verify("copy_type_converted")
        print(f"copy_type_converted: {result.status.value}")
        print(f"  MAE: {result.mae:.2e}, Max Error: {result.max_error:.2e}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        
        # 批量测试
        print("\n🧪 Batch testing...")
        results = await verifier.verify_batch([
            'global_avg_pool',
            'softmax',
            'softmax_opt_64'
        ])
        
        for kernel_id, result in results.items():
            status = "✅" if result.passed else "❌"
            print(f"{status} {kernel_id}: {result.status.value}")
    
    # 运行测试
    asyncio.run(test())
