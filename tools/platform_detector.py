#!/usr/bin/env python3
"""
Platform Capability Detector
平台能力检测模块

检测SYCL和CUDA平台支持的数据类型和特性
"""

import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class PlatformCapabilities:
    """平台能力信息"""
    device_name: str
    vendor: str
    float32: bool = True  # 总是支持
    float16: bool = False
    bfloat16: bool = False
    sm_version: Optional[int] = None  # CUDA only
    extensions: list = field(default_factory=list)


class PlatformDetector:
    """平台能力检测器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self._cache = {}
    
    def detect_sycl_capabilities(self, force_refresh: bool = False) -> PlatformCapabilities:
        """
        检测SYCL设备能力
        
        Returns:
            PlatformCapabilities对象
        """
        cache_key = "sycl"
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 创建检测程序
        detect_code = '''
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    try {
        sycl::queue q;
        auto device = q.get_device();
        
        std::cout << "=== SYCL Platform Detection ===" << std::endl;
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
        
        // 检查扩展
        auto extensions = device.get_info<sycl::info::device::extensions>();
        bool has_bf16 = false;
        bool has_fp16 = false;
        
        std::cout << "Extensions:" << std::endl;
        for (const auto& ext : extensions) {
            std::cout << "  " << ext << std::endl;
            if (ext.find("bfloat16") != std::string::npos) has_bf16 = true;
            if (ext.find("fp16") != std::string::npos || 
                ext.find("half") != std::string::npos) has_fp16 = true;
        }
        
        std::cout << "BF16_EXT: " << (has_bf16 ? "YES" : "NO") << std::endl;
        std::cout << "FP16_EXT: " << (has_fp16 ? "YES" : "NO") << std::endl;
        
        // 测试编译bf16
        #if defined(SYCL_EXT_ONEAPI_BFLOAT16)
        std::cout << "BF16_MACRO: YES" << std::endl;
        #else
        std::cout << "BF16_MACRO: NO" << std::endl;
        #endif
        
        // 尝试使用bf16
        #ifdef SYCL_EXT_ONEAPI_BFLOAT16
        try {
            sycl::ext::oneapi::bfloat16 test_val(1.0f);
            std::cout << "BF16_USABLE: YES" << std::endl;
        } catch (...) {
            std::cout << "BF16_USABLE: NO" << std::endl;
        }
        #else
        std::cout << "BF16_USABLE: NO" << std::endl;
        #endif
        
        // 尝试使用half/fp16
        try {
            sycl::half test_val(1.0f);
            std::cout << "FP16_USABLE: YES" << std::endl;
        } catch (...) {
            std::cout << "FP16_USABLE: NO" << std::endl;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
'''
        
            # 使用docker容器编译和运行检测程序
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(detect_code)
            local_cpp = f.name
        
        try:
            # 复制到docker容器
            container_cpp = "/workspace/detect_sycl.cpp"
            subprocess.run(
                ['docker', 'cp', local_cpp, f'lsv-container:{container_cpp}'],
                capture_output=True, timeout=30
            )
            
            # 在容器内编译
            compile_cmd = [
                'docker', 'exec', 'lsv-container', 'bash', '-c',
                f'cd /workspace && icpx -fsycl -O2 -std=c++17 {container_cpp} -o detect_sycl'
            ]
            result = subprocess.run(compile_cmd, capture_output=True, 
                                  text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"SYCL detection compilation failed: {result.stderr[:500]}")
                return PlatformCapabilities(
                    device_name="Unknown",
                    vendor="Unknown",
                    float32=True,
                    float16=False,
                    bfloat16=False
                )
            
            # 在容器内运行
            run_cmd = ['docker', 'exec', 'lsv-container', '/workspace/detect_sycl']
            result = subprocess.run(run_cmd, capture_output=True, 
                                  text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"SYCL detection execution failed: {result.stderr[:500]}")
                return PlatformCapabilities(
                    device_name="Unknown",
                    vendor="Unknown",
                    float32=True,
                    float16=False,
                    bfloat16=False
                )
            
            # 解析输出
            caps = self._parse_sycl_output(result.stdout)
            self._cache[cache_key] = caps
            return caps
            
        except Exception as e:
            print(f"SYCL detection error: {e}")
            return PlatformCapabilities(
                device_name="Unknown",
                vendor="Unknown",
                float32=True,
                float16=False,
                bfloat16=False
            )
        finally:
            import os
            try:
                os.unlink(local_cpp)
            except:
                pass
    
    def _parse_sycl_output(self, output: str) -> PlatformCapabilities:
        """解析SYCL检测输出"""
        lines = output.strip().split('\n')
        
        device_name = "Unknown"
        vendor = "Unknown"
        bf16_usable = False
        fp16_usable = False
        extensions = []
        
        in_extensions = False
        for line in lines:
            line = line.strip()
            if line.startswith('Device:'):
                device_name = line.split(':', 1)[1].strip()
            elif line.startswith('Vendor:'):
                vendor = line.split(':', 1)[1].strip()
            elif line.startswith('BF16_USABLE:'):
                bf16_usable = 'YES' in line
            elif line.startswith('FP16_USABLE:'):
                fp16_usable = 'YES' in line
            elif line == 'Extensions:':
                in_extensions = True
            elif in_extensions and line.startswith('  '):
                ext = line.strip()
                if ext:
                    extensions.append(ext)
            elif in_extensions and not line.startswith('  '):
                in_extensions = False
        
        return PlatformCapabilities(
            device_name=device_name,
            vendor=vendor,
            float32=True,  # 总是支持
            float16=fp16_usable,
            bfloat16=bf16_usable,
            extensions=extensions
        )
    
    def detect_cuda_capabilities(self, force_refresh: bool = False) -> PlatformCapabilities:
        """
        检测CUDA设备能力
        
        Returns:
            PlatformCapabilities对象
        """
        cache_key = "cuda"
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 使用远程CUDA环境
        remote_host = "root@10.112.229.160"
        
        detect_code = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "=== CUDA Platform Detection ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Vendor: NVIDIA" << std::endl;
    std::cout << "SM_Version: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM_Number: " << (prop.major * 10 + prop.minor) << std::endl;
    
    // 检查bf16支持 (SM80+)
    bool bf16_support = (prop.major > 8) || (prop.major == 8 && prop.minor >= 0);
    std::cout << "BF16_SUPPORT: " << (bf16_support ? "YES" : "NO") << std::endl;
    
    // 检查fp16支持 (SM53+)
    bool fp16_support = (prop.major > 5) || (prop.major == 5 && prop.minor >= 3);
    std::cout << "FP16_SUPPORT: " << (fp16_support ? "YES" : "NO") << std::endl;
    
    // 尝试编译bf16代码
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    std::cout << "BF16_NATIVE: YES" << std::endl;
    #else
    std::cout << "BF16_NATIVE: CHECK_RUNTIME" << std::endl;
    #endif
    
    return 0;
}
'''
        
        local_cu = None
        try:
            # 写入远程主机
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', 
                                            delete=False) as f:
                f.write(detect_code)
                local_cu = f.name
            
            # 先复制到远程主机，再复制到docker容器
            remote_cu = "/tmp/detect_cuda.cu"
            scp_result = subprocess.run(
                ['scp', local_cu, f'{remote_host}:{remote_cu}'],
                capture_output=True, text=True, timeout=30
            )
            
            if scp_result.returncode != 0:
                print(f"Failed to copy to remote host: {scp_result.stderr}")
                return self._default_cuda_caps()
            
            # 从远程主机复制到docker容器
            docker_cp_cmd = f'ssh {remote_host} "docker cp {remote_cu} cuda12.9-test:/workspace/detect_cuda.cu"'
            cp_result = subprocess.run(docker_cp_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if cp_result.returncode != 0:
                print(f"Failed to copy to docker container: {cp_result.stderr}")
                return self._default_cuda_caps()
            
            # 编译并运行
            ssh_cmd = f'''
            ssh {remote_host} '
            docker exec cuda12.9-test bash -c "
            cd /workspace && 
            nvcc -O2 detect_cuda.cu -o detect_cuda && 
            ./detect_cuda
            "
            '
            '''
            
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True,
                                  text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"CUDA detection failed: {result.stderr}")
                return self._default_cuda_caps()
            
            # 解析输出
            caps = self._parse_cuda_output(result.stdout)
            self._cache[cache_key] = caps
            return caps
            
        except Exception as e:
            print(f"CUDA detection error: {e}")
            return self._default_cuda_caps()
        finally:
            import os
            if local_cu is not None:
                try:
                    os.unlink(local_cu)
                except:
                    pass
    
    def _parse_cuda_output(self, output: str) -> PlatformCapabilities:
        """解析CUDA检测输出"""
        lines = output.strip().split('\n')
        
        device_name = "Unknown"
        sm_version = 0
        bf16_support = False
        fp16_support = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Device:'):
                device_name = line.split(':', 1)[1].strip()
            elif line.startswith('SM_Number:'):
                try:
                    sm_version = int(line.split(':', 1)[1].strip())
                except:
                    pass
            elif line.startswith('BF16_SUPPORT:'):
                bf16_support = 'YES' in line
            elif line.startswith('FP16_SUPPORT:'):
                fp16_support = 'YES' in line
        
        return PlatformCapabilities(
            device_name=device_name,
            vendor="NVIDIA",
            float32=True,
            float16=fp16_support,
            bfloat16=bf16_support,
            sm_version=sm_version
        )
    
    def _default_cuda_caps(self) -> PlatformCapabilities:
        """默认CUDA能力（检测失败时）"""
        return PlatformCapabilities(
            device_name="Unknown",
            vendor="NVIDIA",
            float32=True,
            float16=False,
            bfloat16=False
        )
    
    def get_capabilities_summary(self) -> Dict:
        """获取能力摘要"""
        sycl_caps = self.detect_sycl_capabilities()
        cuda_caps = self.detect_cuda_capabilities()
        
        return {
            "sycl": {
                "device": sycl_caps.device_name,
                "vendor": sycl_caps.vendor,
                "float32": sycl_caps.float32,
                "float16": sycl_caps.float16,
                "bfloat16": sycl_caps.bfloat16,
                "extensions": sycl_caps.extensions[:5]  # 只显示前5个
            },
            "cuda": {
                "device": cuda_caps.device_name,
                "vendor": cuda_caps.vendor,
                "sm_version": cuda_caps.sm_version,
                "float32": cuda_caps.float32,
                "float16": cuda_caps.float16,
                "bfloat16": cuda_caps.bfloat16
            },
            "common_support": {
                "float32": True,
                "float16": sycl_caps.float16 and cuda_caps.float16,
                "bfloat16": sycl_caps.bfloat16 and cuda_caps.bfloat16
            }
        }


# 便捷函数
def detect_platforms() -> Dict:
    """检测所有平台能力"""
    detector = PlatformDetector()
    return detector.get_capabilities_summary()


if __name__ == "__main__":
    import sys
    print("Detecting platform capabilities...")
    
    detector = PlatformDetector()
    
    print("\n=== SYCL Detection ===")
    sycl_caps = detector.detect_sycl_capabilities()
    print(f"Device: {sycl_caps.device_name}")
    print(f"Vendor: {sycl_caps.vendor}")
    print(f"Float16: {'YES' if sycl_caps.float16 else 'NO'}")
    print(f"BFloat16: {'YES' if sycl_caps.bfloat16 else 'NO'}")
    
    print("\n=== CUDA Detection ===")
    cuda_caps = detector.detect_cuda_capabilities()
    print(f"Device: {cuda_caps.device_name}")
    print(f"Vendor: {cuda_caps.vendor}")
    print(f"SM Version: {cuda_caps.sm_version}")
    print(f"Float16: {'YES' if cuda_caps.float16 else 'NO'}")
    print(f"BFloat16: {'YES' if cuda_caps.bfloat16 else 'NO'}")
    
    print("\n=== Summary ===")
    summary = detector.get_capabilities_summary()
    print(json.dumps(summary, indent=2))
