# Makefile for CUDA test programs
# Tests for LCZero kernels

# Compilers
NVCC = nvcc
ICPX = icpx

# CUDA flags
NVCC_FLAGS = -O2 -Wno-deprecated-gpu-targets -arch=sm_89
CUDA_LIBS = -lcudart

# SYCL flags
SYCL_FLAGS = -fsycl -O2

# Directories
SRC_DIR = src
INC_DIR = include
REF_DIR = reference_data

# Test programs
CUDA_TESTS = test_batch_norm test_global_avg_pool test_policy_map test_softmax
SYCL_TESTS = test_batch_norm_sycl test_global_avg_pool_sycl test_policy_map_sycl test_softmax_sycl

# Default target
all: cuda

cuda: $(CUDA_TESTS)

sycl: $(SYCL_TESTS)

# CUDA test programs
test_batch_norm: $(SRC_DIR)/test_batch_norm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIBS)

test_global_avg_pool: $(SRC_DIR)/test_global_avg_pool.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIBS)

test_policy_map: $(SRC_DIR)/test_policy_map.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIBS)

test_softmax: $(SRC_DIR)/test_softmax.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIBS)

# SYCL test programs
test_batch_norm_sycl: $(SRC_DIR)/test_batch_norm_sycl.cpp
	$(ICPX) $(SYCL_FLAGS) -o $@ $<

test_global_avg_pool_sycl: $(SRC_DIR)/test_global_avg_pool_sycl.cpp
	$(ICPX) $(SYCL_FLAGS) -o $@ $<

test_policy_map_sycl: $(SRC_DIR)/test_policy_map_sycl.cpp
	$(ICPX) $(SYCL_FLAGS) -o $@ $<

test_softmax_sycl: $(SRC_DIR)/test_softmax_sycl.cpp
	$(ICPX) $(SYCL_FLAGS) -o $@ $<

# Run tests
run: all
	@echo "Running CUDA tests..."
	@for test in $(CUDA_TESTS); do \
		echo "Running $$test..."; \
		./$$test || echo "$$test failed"; \
	done

run_sycl: sycl
	@echo "Running SYCL tests..."
	@for test in $(SYCL_TESTS); do \
		echo "Running $$test..."; \
		./$$test || echo "$$test failed"; \
	done

# Clean
clean:
	rm -f $(CUDA_TESTS) $(SYCL_TESTS)
	rm -f $(REF_DIR)/*.bin

.PHONY: all cuda sycl run run_sycl clean
