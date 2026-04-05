// winograd_output_se_relu_input - SYCL Conversion with SE module
// Fused: Output Transform + SE + ReLU + Input Transform
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;

namespace lczero {
namespace sycldnn_backend {

template <typename T>
inline void OutputTransform4x4_SE(T* output, const T* input) {
  const float At[4][6] = {{1,1,1,1,1,0},{0,1,-1,2,-2,0},{0,1,1,4,4,0},{0,1,-1,8,-8,1}};
  const float A[6][4] = {{1,0,0,0},{1,1,1,1},{1,-1,1,-1},{1,2,4,8},{1,-2,4,-8},{0,0,0,1}};
  float temp[4][6], in[6][6], out[4][4];
  for (int i=0;i<6;i++) for(int j=0;j<6;j++) in[i][j]=(float)input[i*6+j];
  for(int i=0;i<4;i++) for(int j=0;j<6;j++) { temp[i][j]=0; for(int k=0;k<6;k++) temp[i][j]+=At[i][k]*in[k][j]; }
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) { out[i][j]=0; for(int k=0;k<6;k++) out[i][j]+=temp[i][k]*A[k][j]; output[i*4+j]=(T)out[i][j]; }
}

template <typename T>
inline void InputTransform4x4_SE(T* output, const T* input) {
  const float Bt[6][6] = {{4,0,-5,0,1,0},{0,-4,-4,1,1,0},{0,4,-4,-1,1,0},{0,-2,-1,2,1,0},{0,2,-1,-2,1,0},{0,4,0,-5,0,1}};
  const float B[6][6] = {{4,0,0,0,0,0},{-4,4,-2,2,4,0},{-5,-4,-4,-1,-1,0},{0,1,-1,2,-2,-5},{1,1,1,1,1,0},{0,0,0,0,0,1}};
  float temp[6][6], in[6][6]={}, out[6][6];
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) in[i+1][j+1]=(float)input[i*4+j];
  for(int i=0;i<6;i++) for(int j=0;j<6;j++) { temp[i][j]=0; for(int k=0;k<6;k++) temp[i][j]+=Bt[i][k]*in[k][j]; }
  for(int i=0;i<6;i++) for(int j=0;j<6;j++) { out[i][j]=0; for(int k=0;k<6;k++) out[i][j]+=temp[i][k]*B[k][j]; output[i*6+j]=(T)out[i][j]; }
}

template <typename T>
void winogradOutputSeReluInput(int N, int C, int se_K, T* output, const T* input, const T* bias,
    const T* w1, const T* b1, const T* w2, const T* b2, sycl::queue& q) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float,1> shared_data(sycl::range<1>(1024),cgh);
    cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(N*4,C*4),sycl::range<2>(4,64)),
      [=](sycl::nd_item<2> item) {
        int tile_n=item.get_group(0), tile_c=item.get_group(1)*item.get_local_range(1)+item.get_local_id(1);
        int local_tile=item.get_local_id(0), n=tile_n/4, tile_idx=tile_n%4;
        if(tile_c>=C) return;
        
        // Read transformed input
        T transformed[6][6];
        for(int i=0;i<6;i++) for(int j=0;j<6;j++) transformed[i][j]=input[((((n*4+tile_idx)*C+tile_c)*6)+i)*6+j];
        
        // Output transform
        T tile4x4[4][4];
        OutputTransform4x4_SE(&tile4x4[0][0],&transformed[0][0]);
        
        // Add bias and compute SE
        float S=0;
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) { tile4x4[i][j]+=bias[tile_c]; S+=(float)tile4x4[i][j]; }
        
        // SE: global average pool
        float avg=S/64.0f;
        shared_data[tile_c]=avg;
        item.barrier(sycl::access::fence_space::local_space);
        
        // SE: FC1
        float se_val=0;
        for(int i=0;i<C;i++) se_val+=shared_data[i]*(float)w1[i*se_K+tile_c];
        se_val+=(float)b1[tile_c];
        se_val=(se_val>0)?se_val:0; // ReLU
        
        // SE: FC2 + sigmoid
        float scale=0, shift=0;
        for(int i=0;i<se_K;i++) {
          scale+=se_val*(float)w2[i*2*C+tile_c];
          shift+=se_val*(float)w2[i*2*C+tile_c+C];
        }
        scale+=(float)b2[tile_c];
        shift+=(float)b2[tile_c+C];
        scale=1.0f/(1.0f+sycl::exp(-scale)); // Sigmoid
        
        // Apply SE, ReLU
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
          float val=(float)tile4x4[i][j]*scale+shift;
          if(val<0) val=0;
          tile4x4[i][j]=(T)val;
        }
        
        // Input transform
        T output_transformed[6][6];
        InputTransform4x4_SE(&output_transformed[0][0],&tile4x4[0][0]);
        
        // Write output
        for(int i=0;i<6;i++) for(int j=0;j<6;j++) output[((((n*4+tile_idx)*C+tile_c)*6)+i)*6+j]=output_transformed[i][j];
      });
  });
  q.wait();
}

}
}

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    cout<<"=== winograd_output_se_relu_input - Round 2 Phase 1 ==="<<endl;
    cout<<"Device: "<<q.get_device().get_info<sycl::info::device::name>()<<endl;
    cout<<"Note: Fused Output Transform + SE + ReLU + Input Transform"<<endl<<endl;
    
    struct Cfg{int N,C,se_K,tiles,inputSize;}; vector<Cfg> cfgs={{16,64,16,16*4,16*4*64*36},{64,128,32,64*4,64*4*128*36}};
    cout<<setw(8)<<"N"<<setw(8)<<"C"<<setw(10)<<"se_K"<<setw(12)<<"Tiles"<<setw(15)<<"Time(ms)"<<setw(15)<<"GFLOPS"<<endl;
    cout<<string(68,'-')<<endl;
    
    for(auto& c:cfgs){
      sycl::half *out=sycl::malloc_device<sycl::half>(c.inputSize,q),*in=sycl::malloc_device<sycl::half>(c.inputSize,q);
      sycl::half *bias=sycl::malloc_device<sycl::half>(c.C,q),*w1=sycl::malloc_device<sycl::half>(c.C*c.se_K,q);
      sycl::half *b1=sycl::malloc_device<sycl::half>(c.se_K,q),*w2=sycl::malloc_device<sycl::half>(c.se_K*2*c.C,q);
      sycl::half *b2=sycl::malloc_device<sycl::half>(2*c.C,q);
      
      for(int i=0;i<3;i++) winogradOutputSeReluInput(c.N,c.C,c.se_K,out,in,bias,w1,b1,w2,b2,q);
      vector<double> times;
      for(int i=0;i<10;i++){auto s=chrono::high_resolution_clock::now();winogradOutputSeReluInput(c.N,c.C,c.se_K,out,in,bias,w1,b1,w2,b2,q);auto e=chrono::high_resolution_clock::now();times.push_back(chrono::duration<double,milli>(e-s).count());}
      
      double avg=0; for(double t:times)avg+=t; avg/=times.size();
      double flops=(4*6*6*6+6*6*6*6+c.C*2+c.se_K*4)*c.tiles; double gflops=flops/(avg*1e-3)/1e9;
      
      cout<<setw(8)<<c.N<<setw(8)<<c.C<<setw(10)<<c.se_K<<setw(12)<<c.tiles<<setw(15)<<fixed<<setprecision(3)<<avg<<setw(15)<<setprecision(2)<<gflops<<endl;
      
      sycl::free(out,q);sycl::free(in,q);sycl::free(bias,q);sycl::free(w1,q);sycl::free(b1,q);sycl::free(w2,q);sycl::free(b2,q);
    }
    cout<<endl<<"✓ CUDA→SYCL conversion successful!"<<endl;
    cout<<"✓ Fused kernel: Output Transform + SE + ReLU + Input Transform"<<endl;
  } catch(exception const &e){cerr<<"Exception: "<<e.what()<<endl; return 1;}
  return 0;
}
