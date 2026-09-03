//! darkforest-cuda: CUDA kernel bindings for Dark Forest.
//!
//! This crate provides:
//!   - Memory management: cudaMalloc/cudaFree wrappers
//!   - Host<->Device transfers
//!   - cuBLAS sgemm binding
//!   - Elementwise CUDA kernels
//!   - Fused attention kernel (FlashAttention-2 style, sm_120)
//!
//! When compiled without CUDA toolkit, all functions return Err("CUDA not available").

pub mod kernels;
pub mod memory;

use anyhow::{anyhow, Result};
use std::sync::Arc;

/// Lock-free cuBLAS handle storage.
/// The handle is initialized once via OnceLock (no allocation after init),
/// then accessed via a raw atomic pointer — zero mutex overhead per GEMM call.
#[cfg(darkforest_cuda_kernels)]
struct CublasState {
    handle: *mut std::ffi::c_void,
    #[allow(dead_code)]
    workspace: *mut std::ffi::c_void,
}

#[cfg(darkforest_cuda_kernels)]
unsafe impl Send for CublasState {}
#[cfg(darkforest_cuda_kernels)]
unsafe impl Sync for CublasState {}

#[cfg(darkforest_cuda_kernels)]
fn get_cublas_handle() -> Result<*mut std::ffi::c_void> {
    // OnceLock: initialized once, then accessed with a single atomic load.
    // No Mutex is held during GEMM calls — eliminates ~330ms/step of lock overhead.
    static CUBLAS_STATE: std::sync::OnceLock<CublasState> = std::sync::OnceLock::new();

    let state = CUBLAS_STATE.get_or_init(|| {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe { cublas::cublasCreate_v2(&mut handle) };
        assert!(status == 0, "cublasCreate_v2 failed with code {status}");
        // Enable Tensor Core Math (TF32 / Fast FP32 MMA on Ampere+, Blackwell)
        unsafe {
            cublas::cublasSetMathMode(handle, cublas::CUBLAS_TF32_TENSOR_OP_MATH);
        }
        // Pre-allocate 32 MB workspace so cuBLAS never mallocs during graph capture
        const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
        let mut workspace: *mut std::ffi::c_void = std::ptr::null_mut();
        let ws_status = unsafe { cuda_alloc::cudaMalloc(&mut workspace, WORKSPACE_BYTES) };
        assert!(
            ws_status == 0,
            "cudaMalloc for cuBLAS workspace failed: {ws_status}"
        );
        let set_ws_status =
            unsafe { cublas::cublasSetWorkspace_v2(handle, workspace, WORKSPACE_BYTES) };
        assert!(
            set_ws_status == 0,
            "cublasSetWorkspace_v2 failed: {set_ws_status}"
        );
        CublasState { handle, workspace }
    });

    Ok(state.handle)
}

#[cfg(darkforest_cuda_kernels)]
mod cublas {
    use std::ffi::c_void;

    pub const CUBLAS_OP_N: i32 = 0;
    pub const CUBLAS_OP_T: i32 = 1;
    pub const CUBLAS_TF32_TENSOR_OP_MATH: i32 = 1;

    extern "C" {
        pub fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
        pub fn cublasSetMathMode(handle: *mut c_void, mode: i32) -> i32;
        pub fn cublasSetStream_v2(handle: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cublasSetWorkspace_v2(
            handle: *mut c_void,
            workspace: *mut c_void,
            workspace_size_in_bytes: usize,
        ) -> i32;
        pub fn cublasDestroy_v2(handle: *mut c_void) -> i32;
        pub fn cublasSgemm_v2(
            handle: *mut c_void,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: *const f32,
            c: *mut f32,
            ldc: i32,
        ) -> i32;
        // cublasGemmEx — used for BF16 Tensor Core GEMMs
        pub fn cublasGemmEx(
            handle: *mut c_void,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const c_void,
            a: *const c_void,
            atype: i32,
            lda: i32,
            b: *const c_void,
            btype: i32,
            ldb: i32,
            beta: *const c_void,
            c: *mut c_void,
            ctype: i32,
            ldc: i32,
            compute_type: u32,
            algo: i32,
        ) -> i32;
    }

    // CUDA data type constants
    pub const CUDA_R_32F: i32 = 0;
    pub const CUDA_R_16BF: i32 = 14;
    // Compute type: CUBLAS_COMPUTE_32F
    pub const CUBLAS_COMPUTE_32F: u32 = 68;
    // Algorithm: CUBLAS_GEMM_DEFAULT_TENSOR_OP
    pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: i32 = -1;
}

#[cfg(darkforest_cuda_kernels)]
mod cuda_alloc {
    use std::ffi::c_void;
    extern "C" {
        pub fn cudaMalloc(devptr: *mut *mut c_void, size: usize) -> i32;
    }
}

#[cfg(darkforest_cuda_kernels)]
mod cuda_event {
    use std::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreate(event: *mut *mut c_void) -> i32;
        pub fn cudaEventDestroy(event: *mut c_void) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(event: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    }
}

pub struct CudaEventTimer {
    #[cfg(darkforest_cuda_kernels)]
    start: *mut std::ffi::c_void,
    #[cfg(darkforest_cuda_kernels)]
    end: *mut std::ffi::c_void,
}

unsafe impl Send for CudaEventTimer {}
unsafe impl Sync for CudaEventTimer {}

impl CudaEventTimer {
    pub fn new() -> Result<Self> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let mut start: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut end: *mut std::ffi::c_void = std::ptr::null_mut();
            let s1 = cuda_event::cudaEventCreate(&mut start);
            let s2 = cuda_event::cudaEventCreate(&mut end);
            if s1 != 0 || s2 != 0 {
                return Err(anyhow!("cudaEventCreate failed"));
            }
            Ok(Self { start, end })
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Ok(Self {})
    }

    pub fn start(&self) -> Result<()> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let s = cuda_event::cudaEventRecord(self.start, std::ptr::null_mut());
            if s != 0 {
                return Err(anyhow!("cudaEventRecord start failed"));
            }
        }
        Ok(())
    }

    pub fn stop_and_elapsed_ms(&self) -> Result<f64> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let s = cuda_event::cudaEventRecord(self.end, std::ptr::null_mut());
            if s != 0 {
                return Err(anyhow!("cudaEventRecord end failed"));
            }
            cuda_event::cudaEventSynchronize(self.end);
            let mut ms: f32 = 0.0;
            cuda_event::cudaEventElapsedTime(&mut ms, self.start, self.end);
            return Ok(ms as f64);
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Ok(0.0)
    }
}

impl Drop for CudaEventTimer {
    fn drop(&mut self) {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            if !self.start.is_null() {
                cuda_event::cudaEventDestroy(self.start);
            }
            if !self.end.is_null() {
                cuda_event::cudaEventDestroy(self.end);
            }
        }
    }
}

#[cfg(darkforest_cuda_kernels)]
mod cuda_graph {
    use std::ffi::c_void;
    extern "C" {
        pub fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
        pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        pub fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
        pub fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
        pub fn cudaGraphInstantiate(
            graph_exec: *mut *mut c_void,
            graph: *mut c_void,
            error_node: *mut c_void,
            log_buffer: *mut u8,
            buffer_size: usize,
        ) -> i32;
        pub fn cudaGraphLaunch(graph_exec: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaGraphExecDestroy(graph_exec: *mut c_void) -> i32;
        pub fn cudaGraphDestroy(graph: *mut c_void) -> i32;
    }
}

pub struct CudaGraphExec {
    #[cfg(darkforest_cuda_kernels)]
    exec: *mut std::ffi::c_void,
    #[cfg(darkforest_cuda_kernels)]
    stream: *mut std::ffi::c_void,
}

unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl CudaGraphExec {
    pub fn capture<F: FnOnce() -> Result<()>>(f: F) -> Result<Self> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
            let s_create = cuda_graph::cudaStreamCreate(&mut stream);
            if s_create != 0 {
                return Err(anyhow!("cudaStreamCreate failed: {s_create}"));
            }

            // Set cuBLAS stream to the capturing stream BEFORE beginning capture
            // cuBLAS API calls like cublasSetStream_v2 are not allowed inside a graph capture
            let cublas_handle = get_cublas_handle()?;
            cublas::cublasSetStream_v2(cublas_handle, stream);

            // Mode 0 = cudaStreamCaptureModeGlobal
            let s_begin = cuda_graph::cudaStreamBeginCapture(stream, 0);
            if s_begin != 0 {
                cublas::cublasSetStream_v2(cublas_handle, std::ptr::null_mut());
                cuda_graph::cudaStreamDestroy(stream);
                return Err(anyhow!("cudaStreamBeginCapture failed: {s_begin}"));
            }

            // Run operations during stream capture
            let result = f();

            // Reset cuBLAS stream to default
            cublas::cublasSetStream_v2(cublas_handle, std::ptr::null_mut());

            if let Err(e) = result {
                let mut discard_graph: *mut std::ffi::c_void = std::ptr::null_mut();
                let _ = cuda_graph::cudaStreamEndCapture(stream, &mut discard_graph);
                if !discard_graph.is_null() {
                    let _ = cuda_graph::cudaGraphDestroy(discard_graph);
                }
                cuda_graph::cudaStreamDestroy(stream);
                return Err(e);
            }

            let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
            let s_end = cuda_graph::cudaStreamEndCapture(stream, &mut graph);
            if s_end != 0 || graph.is_null() {
                cuda_graph::cudaStreamDestroy(stream);
                return Err(anyhow!("cudaStreamEndCapture failed: {s_end}"));
            }

            let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
            let s_inst = cuda_graph::cudaGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            cuda_graph::cudaGraphDestroy(graph); // graph template can be destroyed after instantiation
            if s_inst != 0 || exec.is_null() {
                cuda_graph::cudaStreamDestroy(stream);
                return Err(anyhow!("cudaGraphInstantiate failed: {s_inst}"));
            }

            Ok(Self { exec, stream })
        }
        #[cfg(not(darkforest_cuda_kernels))]
        {
            f()?;
            Ok(Self {})
        }
    }

    pub fn launch(&self) -> Result<()> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let s = cuda_graph::cudaGraphLaunch(self.exec, self.stream);
            if s != 0 {
                return Err(anyhow!("cudaGraphLaunch failed: {s}"));
            }
            let sync = cuda_graph::cudaStreamSynchronize(self.stream);
            if sync != 0 {
                return Err(anyhow!("cudaStreamSynchronize failed: {sync}"));
            }
        }
        Ok(())
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            if !self.exec.is_null() {
                cuda_graph::cudaGraphExecDestroy(self.exec);
            }
            if !self.stream.is_null() {
                cuda_graph::cudaStreamDestroy(self.stream);
            }
        }
    }
}

#[derive(Debug)]
pub struct DeviceTensor {
    pub shape: Vec<usize>,
    buffer: memory::CudaBuffer,
}

pub struct AdamWState {
    parameter: DeviceTensor,
    first_moment: DeviceTensor,
    second_moment: DeviceTensor,
    gradient: DeviceTensor,
    step: usize,
}

impl AdamWState {
    pub fn new(parameter: &[f32], shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let parameter = DeviceTensor::from_host(parameter, shape.clone())?;
        let first_moment = DeviceTensor::zeros(shape.clone())?;
        let second_moment = DeviceTensor::zeros(shape.clone())?;
        let gradient = DeviceTensor::zeros(shape)?;
        Ok(Self {
            parameter,
            first_moment,
            second_moment,
            gradient,
            step: 0,
        })
    }

    pub fn step(
        &mut self,
        gradient: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<()> {
        if gradient.len() != self.parameter.numel() {
            return Err(anyhow!("gradient length does not match AdamW parameter"));
        }
        self.gradient.upload(gradient)?;
        self.step += 1;
        let step = self.step as f32;
        DeviceTensor::adamw_update(
            &self.parameter,
            &self.first_moment,
            &self.second_moment,
            &self.gradient,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            1.0 - beta1.powf(step),
            1.0 - beta2.powf(step),
        )
    }

    pub fn parameter(&self) -> Result<Vec<f32>> {
        self.parameter.download()
    }

    pub fn sync_parameter(&self, destination: &mut Vec<f32>) -> Result<()> {
        *destination = self.parameter.download()?;
        Ok(())
    }

    pub fn step_count(&self) -> usize {
        self.step
    }
}

impl DeviceTensor {
    pub fn from_host(data: &[f32], shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.iter().product::<usize>();
        if data.len() != elements {
            return Err(anyhow!(
                "host data length does not match device tensor shape"
            ));
        }
        let allocator = Arc::new(memory::GpuAllocator);
        let buffer = memory::CudaBuffer::new(elements * std::mem::size_of::<f32>(), allocator)?;
        buffer.upload(data)?;
        Ok(Self { shape, buffer })
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.iter().product::<usize>();
        let allocator = Arc::new(memory::GpuAllocator);
        let buffer = memory::CudaBuffer::zeros(elements * std::mem::size_of::<f32>(), allocator)?;
        Ok(Self { shape, buffer })
    }

    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.iter().product::<usize>();
        let allocator = Arc::new(memory::GpuAllocator);
        let buffer = memory::CudaBuffer::new(elements * std::mem::size_of::<f32>(), allocator)?;
        kernels::cuda_fill(buffer.ptr as *mut f32, 1.0f32, elements)?;
        Ok(Self { shape, buffer })
    }

    pub fn fill(&mut self, val: f32) -> Result<()> {
        let elements = self.numel();
        kernels::cuda_fill(self.as_mut_ptr(), val, elements)
    }

    pub fn reduce_sum(&self) -> Result<DeviceTensor> {
        let out = DeviceTensor::zeros(vec![1])?;
        kernels::cuda_reduce_sum(self.as_ptr(), out.as_mut_ptr(), self.numel())?;
        Ok(out)
    }

    pub fn clone_tensor(&self) -> Result<Self> {
        let allocator = Arc::new(memory::GpuAllocator);
        let buffer = memory::CudaBuffer::new(self.buffer.bytes, allocator)?;
        buffer.copy_from(&self.buffer)?;
        Ok(Self {
            shape: self.shape.clone(),
            buffer,
        })
    }

    pub fn copy_from(&mut self, other: &DeviceTensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(anyhow!("copy_from shape mismatch"));
        }
        self.buffer.copy_from(&other.buffer)
    }

    pub fn zero_(&mut self) -> Result<()> {
        self.buffer.memset_zero()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn download(&self) -> Result<Vec<f32>> {
        let mut data = Vec::new();
        self.buffer.download(&mut data)?;
        Ok(data)
    }

    pub fn async_download_scalar_f32(&self, dst: &memory::PinnedBuffer) -> Result<()> {
        self.buffer.async_download_scalar_f32(dst)
    }

    pub fn upload(&self, data: &[f32]) -> Result<()> {
        if data.len() != self.numel() {
            return Err(anyhow!(
                "host data length does not match device tensor shape"
            ));
        }
        self.buffer.upload(data)
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.buffer.ptr as *const f32
    }

    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.buffer.ptr as *mut f32
    }

    /// Return a raw pointer offset into this buffer by `offset` f32 elements.
    /// Safety: caller must ensure offset + count <= numel().
    pub fn as_ptr_offset(&self, offset: usize) -> *const f32 {
        unsafe { (self.buffer.ptr as *const f32).add(offset) }
    }

    pub fn as_mut_ptr_offset(&self, offset: usize) -> *mut f32 {
        unsafe { (self.buffer.ptr as *mut f32).add(offset) }
    }

    pub fn add(&self, other: &DeviceTensor) -> Result<DeviceTensor> {
        if self.shape != other.shape {
            return Err(anyhow!("DeviceTensor::add shape mismatch"));
        }
        let mut out = DeviceTensor::zeros(self.shape.clone())?;
        self.add_into(other, &mut out)?;
        Ok(out)
    }

    pub fn add_into(&self, other: &DeviceTensor, out: &mut DeviceTensor) -> Result<()> {
        if self.shape != other.shape || self.shape != out.shape {
            return Err(anyhow!("DeviceTensor::add_into shape mismatch"));
        }
        kernels::cuda_add(
            self.as_ptr(),
            other.as_ptr(),
            out.as_mut_ptr(),
            self.numel(),
        )?;
        Ok(())
    }

    pub fn add_inplace(&mut self, other: &DeviceTensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(anyhow!("DeviceTensor::add_inplace shape mismatch"));
        }
        let mut tmp = DeviceTensor::zeros(self.shape.clone())?;
        kernels::cuda_add(
            self.as_ptr(),
            other.as_ptr(),
            tmp.as_mut_ptr(),
            self.numel(),
        )?;
        self.copy_from(&tmp)
    }

    pub fn scale(&self, alpha: f32) -> Result<DeviceTensor> {
        let out = DeviceTensor::zeros(self.shape.clone())?;
        kernels::cuda_scale(self.as_ptr(), out.as_mut_ptr(), alpha, self.numel())?;
        Ok(out)
    }

    pub fn mul(&self, other: &DeviceTensor) -> Result<DeviceTensor> {
        if self.shape != other.shape {
            return Err(anyhow!("DeviceTensor::mul shape mismatch"));
        }
        let out = DeviceTensor::zeros(self.shape.clone())?;
        kernels::cuda_mul(
            self.as_ptr(),
            other.as_ptr(),
            out.as_mut_ptr(),
            self.numel(),
        )?;
        Ok(out)
    }

    pub fn mul_backward(
        &self,
        other: &DeviceTensor,
        grad_out: &DeviceTensor,
    ) -> Result<(DeviceTensor, DeviceTensor)> {
        if self.shape != other.shape || self.shape != grad_out.shape {
            return Err(anyhow!("DeviceTensor::mul_backward shape mismatch"));
        }
        let grad_a = DeviceTensor::zeros(self.shape.clone())?;
        let grad_b = DeviceTensor::zeros(self.shape.clone())?;
        kernels::cuda_mul_backward(
            grad_out.as_ptr(),
            self.as_ptr(),
            other.as_ptr(),
            grad_a.as_mut_ptr(),
            grad_b.as_mut_ptr(),
            self.numel(),
        )?;
        Ok((grad_a, grad_b))
    }

    pub fn gelu(&self) -> Result<DeviceTensor> {
        let out = DeviceTensor::zeros(self.shape.clone())?;
        kernels::cuda_gelu(self.as_ptr(), out.as_mut_ptr(), self.numel())?;
        Ok(out)
    }

    pub fn gelu_into(&self, out: &mut DeviceTensor) -> Result<()> {
        if self.shape != out.shape {
            return Err(anyhow!("DeviceTensor::gelu_into shape mismatch"));
        }
        kernels::cuda_gelu(self.as_ptr(), out.as_mut_ptr(), self.numel())?;
        Ok(())
    }

    pub fn softmax(&self) -> Result<DeviceTensor> {
        let cols = *self
            .shape
            .last()
            .ok_or_else(|| anyhow!("softmax empty shape"))?;
        let rows = self.numel() / cols;
        let out = self.clone_tensor()?;
        kernels::cuda_softmax(out.as_mut_ptr(), rows, cols)?;
        Ok(out)
    }

    pub fn softmax_backward(
        probabilities: &DeviceTensor,
        grad_output: &DeviceTensor,
    ) -> Result<DeviceTensor> {
        if probabilities.shape != grad_output.shape {
            return Err(anyhow!("softmax backward shape mismatch"));
        }
        let cols = *probabilities
            .shape
            .last()
            .ok_or_else(|| anyhow!("softmax empty shape"))?;
        let rows = probabilities.numel() / cols;
        let grad_input = DeviceTensor::zeros(probabilities.shape.clone())?;
        kernels::cuda_softmax_backward(
            probabilities.as_ptr(),
            grad_output.as_ptr(),
            grad_input.as_mut_ptr(),
            rows,
            cols,
        )?;
        Ok(grad_input)
    }

    pub fn cross_entropy(
        logits: &DeviceTensor,
        targets: &[usize],
    ) -> Result<(DeviceTensor, DeviceTensor)> {
        let cols = *logits
            .shape
            .last()
            .ok_or_else(|| anyhow!("cross entropy empty shape"))?;
        let rows = logits.numel() / cols;
        if targets.len() != rows || targets.iter().any(|&target| target >= cols) {
            return Err(anyhow!("cross entropy targets do not match logits"));
        }
        let probabilities = logits.softmax()?;
        let target_data: Vec<u32> = targets.iter().map(|&target| target as u32).collect();
        let allocator = Arc::new(memory::GpuAllocator);
        let target_buffer =
            memory::CudaBuffer::new(target_data.len() * std::mem::size_of::<u32>(), allocator)?;
        target_buffer.upload_bytes(
            target_data.as_ptr() as *const u8,
            target_data.len() * std::mem::size_of::<u32>(),
        )?;
        let loss = DeviceTensor::zeros(vec![1])?;
        kernels::cuda_cross_entropy_forward(
            probabilities.as_ptr(),
            target_buffer.ptr as *const u32,
            loss.as_mut_ptr(),
            rows,
            cols,
        )?;
        Ok((loss, probabilities))
    }

    pub fn cross_entropy_backward(
        probabilities: &DeviceTensor,
        targets: &[usize],
        grad_output: &DeviceTensor,
    ) -> Result<DeviceTensor> {
        let cols = *probabilities
            .shape
            .last()
            .ok_or_else(|| anyhow!("cross entropy empty shape"))?;
        let rows = probabilities.numel() / cols;
        if targets.len() != rows || targets.iter().any(|&target| target >= cols) {
            return Err(anyhow!("cross entropy targets do not match probabilities"));
        }
        if grad_output.numel() != 1 {
            return Err(anyhow!("cross entropy upstream gradient must be scalar"));
        }
        let target_data: Vec<u32> = targets.iter().map(|&target| target as u32).collect();
        let allocator = Arc::new(memory::GpuAllocator);
        let target_buffer =
            memory::CudaBuffer::new(target_data.len() * std::mem::size_of::<u32>(), allocator)?;
        target_buffer.upload_bytes(
            target_data.as_ptr() as *const u8,
            target_data.len() * std::mem::size_of::<u32>(),
        )?;
        let grad_logits = DeviceTensor::zeros(probabilities.shape.clone())?;
        kernels::cuda_cross_entropy_backward(
            probabilities.as_ptr(),
            target_buffer.ptr as *const u32,
            grad_output.as_ptr(),
            grad_logits.as_mut_ptr(),
            rows,
            cols,
        )?;
        Ok(grad_logits)
    }

    pub fn gelu_backward(&self, grad_out: &DeviceTensor) -> Result<DeviceTensor> {
        if self.shape != grad_out.shape {
            return Err(anyhow!("DeviceTensor::gelu_backward shape mismatch"));
        }
        let mut grad_x = DeviceTensor::zeros(self.shape.clone())?;
        self.gelu_backward_into(grad_out, &mut grad_x)?;
        Ok(grad_x)
    }

    pub fn gelu_backward_into(
        &self,
        grad_out: &DeviceTensor,
        out: &mut DeviceTensor,
    ) -> Result<()> {
        if self.shape != grad_out.shape || self.shape != out.shape {
            return Err(anyhow!("DeviceTensor::gelu_backward_into shape mismatch"));
        }
        kernels::cuda_gelu_backward(
            grad_out.as_ptr(),
            self.as_ptr(),
            out.as_mut_ptr(),
            self.numel(),
        )?;
        Ok(())
    }

    pub fn add_bias(&self, bias: &DeviceTensor) -> Result<DeviceTensor> {
        if self.shape.is_empty() || bias.shape.len() != 1 {
            return Err(anyhow!("DeviceTensor::add_bias invalid rank"));
        }
        let features = *self.shape.last().unwrap();
        if bias.shape[0] != features {
            return Err(anyhow!("DeviceTensor::add_bias feature mismatch"));
        }
        let batch = self.numel() / features;
        let out = DeviceTensor::zeros(self.shape.clone())?;
        kernels::cuda_add_bias(
            self.as_ptr(),
            bias.as_ptr(),
            out.as_mut_ptr(),
            batch,
            features,
        )?;
        Ok(out)
    }

    pub fn embedding_lookup(indices: &[usize], weight: &DeviceTensor) -> Result<DeviceTensor> {
        if weight.shape.len() != 2 {
            return Err(anyhow!("embedding weight must be 2D"));
        }
        let seq_len = indices.len();
        let embed_dim = weight.shape[1];
        let mut out = DeviceTensor::zeros(vec![seq_len, embed_dim])?;
        Self::embedding_lookup_into(indices, weight, &mut out)?;
        Ok(out)
    }

    pub fn embedding_lookup_into(
        indices: &[usize],
        weight: &DeviceTensor,
        out: &mut DeviceTensor,
    ) -> Result<()> {
        if weight.shape.len() != 2 || out.shape != vec![indices.len(), weight.shape[1]] {
            return Err(anyhow!("embedding_lookup_into shape mismatch"));
        }
        let u32_indices: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let allocator = Arc::new(memory::GpuAllocator);
        let idx_buf =
            memory::CudaBuffer::new(indices.len() * std::mem::size_of::<u32>(), allocator)?;
        idx_buf.upload_bytes(
            u32_indices.as_ptr() as *const u8,
            indices.len() * std::mem::size_of::<u32>(),
        )?;
        kernels::cuda_embedding_forward(
            idx_buf.ptr as *const u32,
            weight.as_ptr(),
            out.as_mut_ptr(),
            indices.len(),
            weight.shape[1],
        )?;
        Ok(())
    }

    pub fn embedding_backward(
        indices: &[usize],
        grad_out: &DeviceTensor,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<DeviceTensor> {
        let seq_len = indices.len();
        let u32_indices: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let allocator = Arc::new(memory::GpuAllocator);
        let idx_buf = memory::CudaBuffer::new(seq_len * std::mem::size_of::<u32>(), allocator)?;
        idx_buf.upload_bytes(
            u32_indices.as_ptr() as *const u8,
            seq_len * std::mem::size_of::<u32>(),
        )?;
        let grad_weight = DeviceTensor::zeros(vec![vocab_size, embed_dim])?;
        kernels::cuda_embedding_backward(
            idx_buf.ptr as *const u32,
            grad_out.as_ptr(),
            grad_weight.as_mut_ptr(),
            seq_len,
            embed_dim,
        )?;
        Ok(grad_weight)
    }

    pub fn layernorm(
        &self,
        gamma: Option<&DeviceTensor>,
        beta: Option<&DeviceTensor>,
    ) -> Result<(DeviceTensor, DeviceTensor, DeviceTensor)> {
        let features = *self
            .shape
            .last()
            .ok_or_else(|| anyhow!("layernorm empty shape"))?;
        let batch = self.numel() / features;
        let mut out = DeviceTensor::zeros(self.shape.clone())?;
        let mut means = DeviceTensor::zeros(vec![batch])?;
        let mut rstds = DeviceTensor::zeros(vec![batch])?;
        self.layernorm_into(gamma, beta, &mut out, &mut means, &mut rstds)?;
        Ok((out, means, rstds))
    }

    pub fn layernorm_into(
        &self,
        gamma: Option<&DeviceTensor>,
        beta: Option<&DeviceTensor>,
        out: &mut DeviceTensor,
        means: &mut DeviceTensor,
        rstds: &mut DeviceTensor,
    ) -> Result<()> {
        if self.shape != out.shape {
            return Err(anyhow!("layernorm_into output shape mismatch"));
        }
        let features = *self
            .shape
            .last()
            .ok_or_else(|| anyhow!("layernorm empty shape"))?;
        let batch = self.numel() / features;
        if means.shape != vec![batch] || rstds.shape != vec![batch] {
            return Err(anyhow!("layernorm_into stats shape mismatch"));
        }
        kernels::cuda_layernorm(
            self.as_ptr(),
            gamma.map_or(std::ptr::null(), DeviceTensor::as_ptr),
            beta.map_or(std::ptr::null(), DeviceTensor::as_ptr),
            out.as_mut_ptr(),
            means.as_mut_ptr(),
            rstds.as_mut_ptr(),
            batch,
            features,
        )?;
        Ok(())
    }

    pub fn layernorm_backward(
        grad_out: &DeviceTensor,
        x: &DeviceTensor,
        gamma: Option<&DeviceTensor>,
        means: &DeviceTensor,
        rstds: &DeviceTensor,
    ) -> Result<(DeviceTensor, Option<DeviceTensor>, Option<DeviceTensor>)> {
        let features = *x
            .shape
            .last()
            .ok_or_else(|| anyhow!("layernorm empty shape"))?;
        let batch = x.numel() / features;
        let mut grad_x = DeviceTensor::zeros(x.shape.clone())?;
        let mut grad_gamma = if gamma.is_some() {
            Some(DeviceTensor::zeros(vec![features])?)
        } else {
            None
        };
        let mut grad_beta = Some(DeviceTensor::zeros(vec![features])?);
        Self::layernorm_backward_into(
            grad_out,
            x,
            gamma,
            means,
            rstds,
            &mut grad_x,
            grad_gamma.as_mut(),
            grad_beta.as_mut(),
        )?;
        Ok((grad_x, grad_gamma, grad_beta))
    }

    pub fn layernorm_backward_into(
        grad_out: &DeviceTensor,
        x: &DeviceTensor,
        gamma: Option<&DeviceTensor>,
        means: &DeviceTensor,
        rstds: &DeviceTensor,
        grad_x: &mut DeviceTensor,
        mut grad_gamma: Option<&mut DeviceTensor>,
        mut grad_beta: Option<&mut DeviceTensor>,
    ) -> Result<()> {
        let features = *x
            .shape
            .last()
            .ok_or_else(|| anyhow!("layernorm empty shape"))?;
        let batch = x.numel() / features;
        if let Some(ref mut gg) = grad_gamma {
            gg.zero_()?;
        }
        if let Some(ref mut gb) = grad_beta {
            gb.zero_()?;
        }
        kernels::cuda_layernorm_backward(
            grad_out.as_ptr(),
            x.as_ptr(),
            gamma.map_or(std::ptr::null(), DeviceTensor::as_ptr),
            means.as_ptr(),
            rstds.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_gamma
                .as_mut()
                .map_or(std::ptr::null_mut(), |g| g.as_mut_ptr()),
            grad_beta
                .as_mut()
                .map_or(std::ptr::null_mut(), |b| b.as_mut_ptr()),
            batch,
            features,
        )?;
        Ok(())
    }

    pub fn linear(
        &self,
        weight: &DeviceTensor,
        bias: Option<&DeviceTensor>,
    ) -> Result<DeviceTensor> {
        if self.shape.len() != 2 || weight.shape.len() != 2 {
            return Err(anyhow!("linear requires rank-2 tensors"));
        }
        let batch = self.shape[0];
        let in_features = self.shape[1];
        let out_features = weight.shape[0];
        if weight.shape[1] != in_features {
            return Err(anyhow!("linear in_features mismatch"));
        }
        if let Some(b) = bias {
            if b.shape != vec![out_features] {
                return Err(anyhow!("linear bias shape mismatch"));
            }
        }
        let mut output = DeviceTensor::zeros(vec![batch, out_features])?;
        self.linear_into(weight, bias, &mut output)?;
        Ok(output)
    }

    pub fn linear_into(
        &self,
        weight: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        out: &mut DeviceTensor,
    ) -> Result<()> {
        if self.shape.len() != 2
            || weight.shape.len() != 2
            || out.shape != vec![self.shape[0], weight.shape[0]]
        {
            return Err(anyhow!("linear_into shape mismatch"));
        }
        let batch = self.shape[0];
        let in_features = self.shape[1];
        let out_features = weight.shape[0];
        if weight.shape[1] != in_features {
            return Err(anyhow!("linear in_features mismatch"));
        }
        if let Some(b) = bias {
            if b.shape != vec![out_features] {
                return Err(anyhow!("linear bias shape mismatch"));
            }
        }

        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            // Row-major: C [batch, out_features] = X [batch, in_features] @ W^T [in_features, out_features]
            // In col-major memory:
            // C^T = (X @ W^T)^T = W @ X^T
            // W is [out_features, in_features] row-major -> [in_features, out_features] col-major.
            // X is [batch, in_features] row-major -> [in_features, batch] col-major.
            // W^T in col-major is [out_features, in_features].
            // So: C^T [out_features, batch] = (W_col)^T [out_features, in_features] @ (X_col) [in_features, batch]
            let handle = get_cublas_handle()?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let status = cublas::cublasSgemm_v2(
                handle,
                cublas::CUBLAS_OP_T,
                cublas::CUBLAS_OP_N,
                out_features as i32,
                batch as i32,
                in_features as i32,
                &alpha,
                weight.as_ptr(),
                in_features as i32,
                self.as_ptr(),
                in_features as i32,
                &beta,
                out.as_mut_ptr(),
                out_features as i32,
            );
            if status != 0 {
                return Err(anyhow!("cublasSgemm_v2 failed with code {status}"));
            }
            if let Some(b) = bias {
                kernels::cuda_add_bias(
                    out.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    batch,
                    out_features,
                )?;
            }
            return Ok(());
        }

        #[cfg(not(darkforest_cuda_kernels))]
        {
            kernels::cuda_linear_forward(
                self.as_ptr(),
                weight.as_ptr(),
                bias.map_or(std::ptr::null(), DeviceTensor::as_ptr),
                out.as_mut_ptr(),
                batch,
                in_features,
                out_features,
                bias.is_some(),
            )?;
            Ok(())
        }
    }

    /// BF16 mixed-precision linear: X [batch, in] @ W^T [in, out] → out [batch, out].
    ///
    /// Casts X and W to BF16 on the fly, runs cublasGemmEx with CUBLAS_COMPUTE_32F
    /// (Tensor Core + FP32 accumulation), result is FP32. Requires scratch BF16 buffers
    /// for X and W (passed in to avoid per-call allocation).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16_into(
        &self,
        weight: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        out: &mut DeviceTensor,
        x_bf16: &mut memory::CudaBuffer, // scratch: [batch * in_features] bf16
        w_bf16: &mut memory::CudaBuffer, // scratch: [out_features * in_features] bf16
    ) -> Result<()> {
        let in_features = *self
            .shape
            .last()
            .ok_or_else(|| anyhow!("linear_bf16: empty x shape"))?;
        let batch = self.numel() / in_features;
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| anyhow!("linear_bf16: empty weight shape"))?;

        if out.numel() != batch * out_features {
            return Err(anyhow!("linear_bf16: output shape mismatch"));
        }

        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            use std::ffi::c_void;

            // Cast X → BF16
            kernels::cuda_f32_to_bf16(self.as_ptr(), x_bf16.as_bf16_mut_ptr(), self.numel())?;
            // Cast W → BF16
            kernels::cuda_f32_to_bf16(weight.as_ptr(), w_bf16.as_bf16_mut_ptr(), weight.numel())?;

            let handle = get_cublas_handle()?;
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // C [batch, out_f] = X [batch, in_f] @ W^T [in_f, out_f]
            // cublas col-major:  C^T = W @ X^T
            let status = cublas::cublasGemmEx(
                handle,
                cublas::CUBLAS_OP_T,
                cublas::CUBLAS_OP_N,
                out_features as i32,
                batch as i32,
                in_features as i32,
                &alpha as *const f32 as *const c_void,
                w_bf16.as_ptr() as *const c_void,
                cublas::CUDA_R_16BF,
                in_features as i32,
                x_bf16.as_ptr() as *const c_void,
                cublas::CUDA_R_16BF,
                in_features as i32,
                &beta as *const f32 as *const c_void,
                out.as_mut_ptr() as *mut c_void,
                cublas::CUDA_R_32F,
                out_features as i32,
                cublas::CUBLAS_COMPUTE_32F,
                cublas::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != 0 {
                return Err(anyhow!("cublasGemmEx BF16 failed with code {status}"));
            }
            if let Some(b) = bias {
                kernels::cuda_add_bias(
                    out.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    batch,
                    out_features,
                )?;
            }
            return Ok(());
        }

        #[cfg(not(darkforest_cuda_kernels))]
        {
            // Fallback: plain FP32
            self.linear_into(weight, bias, out)
        }
    }

    pub fn matmul(&self, other: &DeviceTensor) -> Result<DeviceTensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(anyhow!("matmul currently requires rank-2 tensors"));
        }
        let m = self.shape[0];
        let k = self.shape[1];
        if other.shape[0] != k {
            return Err(anyhow!("matmul inner dimensions do not match"));
        }
        let n = other.shape[1];
        let output = DeviceTensor::zeros(vec![m, n])?;

        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            // Row-major: C [m, n] = A [m, k] @ B [k, n]
            // In col-major memory:
            // C^T [n, m] = B^T [n, k] @ A^T [k, m]
            // B_col is [n, k] (which is B^T).
            // A_col is [k, m] (which is A^T).
            // So: C^T = B_col @ A_col
            let handle = get_cublas_handle()?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let status = cublas::cublasSgemm_v2(
                handle,
                cublas::CUBLAS_OP_N,
                cublas::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                other.as_ptr(),
                n as i32,
                self.as_ptr(),
                k as i32,
                &beta,
                output.as_mut_ptr(),
                n as i32,
            );
            if status != 0 {
                return Err(anyhow!("cublasSgemm_v2 matmul failed: {status}"));
            }
            return Ok(output);
        }

        #[cfg(not(darkforest_cuda_kernels))]
        {
            kernels::cuda_matmul(self.as_ptr(), other.as_ptr(), output.as_mut_ptr(), m, k, n)?;
            Ok(output)
        }
    }

    pub fn transpose_2d(&self) -> Result<DeviceTensor> {
        if self.shape.len() != 2 {
            return Err(anyhow!("transpose_2d requires a rank-2 tensor"));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let output = DeviceTensor::zeros(vec![cols, rows])?;
        kernels::cuda_transpose(self.as_ptr(), output.as_mut_ptr(), rows, cols)?;
        Ok(output)
    }

    pub fn matmul_backward(
        a: &DeviceTensor,
        b: &DeviceTensor,
        grad_out: &DeviceTensor,
    ) -> Result<(DeviceTensor, DeviceTensor)> {
        if a.shape.len() != 2 || b.shape.len() != 2 || grad_out.shape.len() != 2 {
            return Err(anyhow!("matmul backward requires rank-2 tensors"));
        }
        if a.shape[1] != b.shape[0] || grad_out.shape != vec![a.shape[0], b.shape[1]] {
            return Err(anyhow!("matmul backward tensor shapes do not match"));
        }

        let b_transposed = b.transpose_2d()?;
        let a_transposed = a.transpose_2d()?;
        let grad_a = grad_out.matmul(&b_transposed)?;
        let grad_b = a_transposed.matmul(grad_out)?;
        Ok((grad_a, grad_b))
    }

    pub fn linear_backward(
        x: &DeviceTensor,
        weight: &DeviceTensor,
        grad_out: &DeviceTensor,
        has_bias: bool,
    ) -> Result<(DeviceTensor, DeviceTensor, Option<DeviceTensor>)> {
        if x.shape.len() != 2 || weight.shape.len() != 2 || grad_out.shape.len() != 2 {
            return Err(anyhow!("linear backward requires rank-2 tensors"));
        }
        let batch = x.shape[0];
        let in_features = x.shape[1];
        let out_features = weight.shape[0];
        if weight.shape[1] != in_features || grad_out.shape != vec![batch, out_features] {
            return Err(anyhow!("linear backward tensor shapes do not match"));
        }
        let mut grad_x = DeviceTensor::zeros(vec![batch, in_features])?;
        let mut grad_weight = DeviceTensor::zeros(vec![out_features, in_features])?;
        let mut grad_bias = if has_bias {
            Some(DeviceTensor::zeros(vec![out_features])?)
        } else {
            None
        };
        Self::linear_backward_into(
            x,
            weight,
            grad_out,
            &mut grad_x,
            &mut grad_weight,
            grad_bias.as_mut(),
        )?;
        Ok((grad_x, grad_weight, grad_bias))
    }

    pub fn linear_backward_into(
        x: &DeviceTensor,
        weight: &DeviceTensor,
        grad_out: &DeviceTensor,
        grad_x: &mut DeviceTensor,
        grad_weight: &mut DeviceTensor,
        grad_bias: Option<&mut DeviceTensor>,
    ) -> Result<()> {
        let batch = x.shape[0];
        let in_features = x.shape[1];
        let out_features = weight.shape[0];

        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let handle = get_cublas_handle()?;
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // 1. grad_x = grad_out @ weight  [batch, in_features]
            // Row-major: grad_x [batch, in_features] = grad_out [batch, out_features] @ weight [out_features, in_features]
            // Col-major: grad_x^T [in_features, batch] = weight^T @ grad_out^T
            // weight_col is [in_features, out_features]. weight_col^T is not needed; weight_col [in_features, out_features] is already weight^T!
            // grad_out_col is [out_features, batch] (which is grad_out^T).
            // So: grad_x^T = weight_col @ grad_out_col
            let status = cublas::cublasSgemm_v2(
                handle,
                cublas::CUBLAS_OP_N,
                cublas::CUBLAS_OP_N,
                in_features as i32,
                batch as i32,
                out_features as i32,
                &alpha,
                weight.as_ptr(),
                in_features as i32,
                grad_out.as_ptr(),
                out_features as i32,
                &beta,
                grad_x.as_mut_ptr(),
                in_features as i32,
            );
            if status != 0 {
                return Err(anyhow!("cublas grad_x failed: {status}"));
            }

            // 2. grad_weight = grad_out^T @ x  [out_features, in_features]
            // Row-major: grad_weight [out_features, in_features] = grad_out^T [out_features, batch] @ x [batch, in_features]
            // Col-major: grad_weight^T [in_features, out_features] = x^T [in_features, batch] @ grad_out [batch, out_features]
            // x_col is [in_features, batch].
            // grad_out_col is [out_features, batch] -> (grad_out_col)^T is [batch, out_features].
            // So: grad_weight^T = x_col [in_features, batch] @ (grad_out_col)^T [batch, out_features]
            let status2 = cublas::cublasSgemm_v2(
                handle,
                cublas::CUBLAS_OP_N,
                cublas::CUBLAS_OP_T,
                in_features as i32,
                out_features as i32,
                batch as i32,
                &alpha,
                x.as_ptr(),
                in_features as i32,
                grad_out.as_ptr(),
                out_features as i32,
                &beta,
                grad_weight.as_mut_ptr(),
                in_features as i32,
            );
            if status2 != 0 {
                return Err(anyhow!("cublas grad_weight failed: {status2}"));
            }

            // 3. grad_bias = sum_rows(grad_out)
            if let Some(gb) = grad_bias {
                extern "C" {
                    fn launch_linear_grad_bias(
                        grad_out: *const f32,
                        grad_bias: *mut f32,
                        batch: u32,
                        out_features: u32,
                    );
                }
                launch_linear_grad_bias(
                    grad_out.as_ptr(),
                    gb.as_mut_ptr(),
                    batch as u32,
                    out_features as u32,
                );
            }

            return Ok(());
        }

        #[cfg(not(darkforest_cuda_kernels))]
        {
            let has_bias = grad_bias.is_some();
            kernels::cuda_linear_backward(
                x.as_ptr(),
                weight.as_ptr(),
                grad_out.as_ptr(),
                grad_x.as_mut_ptr(),
                grad_weight.as_mut_ptr(),
                grad_bias.map_or(std::ptr::null_mut(), |b| b.as_mut_ptr()),
                batch,
                in_features,
                out_features,
                has_bias,
            )?;
            Ok(())
        }
    }

    pub fn linear_backward_accumulate_into(
        x: &DeviceTensor,
        weight: &DeviceTensor,
        grad_out: &DeviceTensor,
        grad_x: &mut DeviceTensor,
        grad_weight: &mut DeviceTensor,
        grad_bias: Option<&mut DeviceTensor>,
    ) -> Result<()> {
        let batch = x.shape[0];
        let in_features = x.shape[1];
        let out_features = weight.shape[0];

        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let handle = get_cublas_handle()?;
            let alpha = 1.0f32;
            let beta_acc = 1.0f32;
            let beta_zero = 0.0f32;

            // 1. grad_x += grad_out @ weight  [batch, in_features] with beta=1.0
            let status = cublas::cublasSgemm_v2(
                handle,
                cublas::CUBLAS_OP_N,
                cublas::CUBLAS_OP_N,
                in_features as i32,
                batch as i32,
                out_features as i32,
                &alpha,
                weight.as_ptr(),
                in_features as i32,
                grad_out.as_ptr(),
                out_features as i32,
                &beta_acc,
                grad_x.as_mut_ptr(),
                in_features as i32,
            );
            if status != 0 {
                return Err(anyhow!("cublas grad_x failed: {status}"));
            }

            // 2. grad_weight = grad_out^T @ x
            let status2 = cublas::cublasSgemm_v2(
                handle,
                cublas::CUBLAS_OP_N,
                cublas::CUBLAS_OP_T,
                in_features as i32,
                out_features as i32,
                batch as i32,
                &alpha,
                x.as_ptr(),
                in_features as i32,
                grad_out.as_ptr(),
                out_features as i32,
                &beta_zero,
                grad_weight.as_mut_ptr(),
                in_features as i32,
            );
            if status2 != 0 {
                return Err(anyhow!("cublas grad_weight failed: {status2}"));
            }

            // 3. grad_bias = sum_rows(grad_out)
            if let Some(gb) = grad_bias {
                extern "C" {
                    fn launch_linear_grad_bias(
                        grad_out: *const f32,
                        grad_bias: *mut f32,
                        batch: u32,
                        out_features: u32,
                    );
                }
                launch_linear_grad_bias(
                    grad_out.as_ptr(),
                    gb.as_mut_ptr(),
                    batch as u32,
                    out_features as u32,
                );
            }

            return Ok(());
        }

        #[cfg(not(darkforest_cuda_kernels))]
        {
            let mut tmp = DeviceTensor::zeros(grad_x.shape.clone())?;
            Self::linear_backward_into(x, weight, grad_out, &mut tmp, grad_weight, grad_bias)?;
            grad_x.add_inplace(&tmp)
        }
    }

    pub fn adamw_update(
        parameter: &DeviceTensor,
        first_moment: &DeviceTensor,
        second_moment: &DeviceTensor,
        gradient: &DeviceTensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
    ) -> Result<()> {
        if parameter.shape != first_moment.shape
            || parameter.shape != second_moment.shape
            || parameter.shape != gradient.shape
        {
            return Err(anyhow!("AdamW tensors must have matching shapes"));
        }
        kernels::cuda_adamw_update(
            parameter.as_mut_ptr(),
            first_moment.as_mut_ptr(),
            second_moment.as_mut_ptr(),
            gradient.as_ptr(),
            parameter.numel(),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
    }

    pub fn embedding_lookup_device_indices(
        d_indices: &memory::CudaBuffer,
        seq_len: usize,
        weight: &DeviceTensor,
        out: &mut DeviceTensor,
    ) -> Result<()> {
        kernels::cuda_embedding_forward(
            d_indices.ptr as *const u32,
            weight.as_ptr(),
            out.as_mut_ptr(),
            seq_len,
            weight.shape[1],
        )
    }

    pub fn embedding_backward_device_indices(
        d_indices: &memory::CudaBuffer,
        seq_len: usize,
        grad_out: &DeviceTensor,
        grad_weight: &mut DeviceTensor,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        grad_weight.zero_()?;
        kernels::cuda_embedding_backward(
            d_indices.ptr as *const u32,
            grad_out.as_ptr(),
            grad_weight.as_mut_ptr(),
            seq_len,
            embed_dim,
        )
    }

    pub fn cross_entropy_device_targets(
        logits: &DeviceTensor,
        d_targets: &memory::CudaBuffer,
        probs: &mut DeviceTensor,
        loss: &mut DeviceTensor,
    ) -> Result<()> {
        let cols = *logits
            .shape
            .last()
            .ok_or_else(|| anyhow!("cross entropy empty shape"))?;
        let rows = logits.numel() / cols;
        probs.copy_from(logits)?;
        kernels::cuda_softmax(probs.as_mut_ptr(), rows, cols)?;
        kernels::cuda_cross_entropy_forward(
            probs.as_ptr(),
            d_targets.ptr as *const u32,
            loss.as_mut_ptr(),
            rows,
            cols,
        )?;
        Ok(())
    }

    pub fn cross_entropy_backward_device_targets(
        probs: &DeviceTensor,
        d_targets: &memory::CudaBuffer,
        grad_output: &DeviceTensor,
        grad_logits: &mut DeviceTensor,
    ) -> Result<()> {
        let cols = *probs
            .shape
            .last()
            .ok_or_else(|| anyhow!("cross entropy empty shape"))?;
        let rows = probs.numel() / cols;
        kernels::cuda_cross_entropy_backward(
            probs.as_ptr(),
            d_targets.ptr as *const u32,
            grad_output.as_ptr(),
            grad_logits.as_mut_ptr(),
            rows,
            cols,
        )?;
        Ok(())
    }

    /// Fused: logits → softmax probs + CE loss + grad_logits = (p - y)/N, all in ONE kernel.
    /// logits is overwritten with softmax probs.
    /// grad_logits is written with (prob - one_hot) / rows.
    /// loss is reset to 0 then filled with mean CE.
    pub fn fused_logit_ce_device_targets(
        logits: &mut DeviceTensor,
        d_targets: &memory::CudaBuffer,
        grad_logits: &mut DeviceTensor,
        loss: &mut DeviceTensor,
    ) -> Result<()> {
        let cols = *logits
            .shape
            .last()
            .ok_or_else(|| anyhow!("fused logit_ce empty shape"))?;
        let rows = logits.numel() / cols;
        kernels::cuda_fused_logit_ce(
            logits.as_mut_ptr(),
            d_targets.ptr as *const u32,
            grad_logits.as_mut_ptr(),
            loss.as_mut_ptr(),
            rows,
            cols,
        )?;
        Ok(())
    }
}

pub struct AttentionContext {
    pub n_heads: usize,
    pub seq_len: usize,
    pub d_head: usize,
    pub d_model: usize,
    q: memory::CudaBuffer,
    k: memory::CudaBuffer,
    v: memory::CudaBuffer,
    d_out: memory::CudaBuffer,
    out: memory::CudaBuffer,
    d_q: memory::CudaBuffer,
    d_k: memory::CudaBuffer,
    d_v: memory::CudaBuffer,
    probabilities: memory::CudaBuffer,
}

impl AttentionContext {
    pub fn new(n_heads: usize, seq_len: usize, d_head: usize) -> Result<Self> {
        if d_head == 0 || d_head > 128 {
            return Err(anyhow!("d_head must be in the range 1..=128"));
        }
        let d_model = n_heads
            .checked_mul(d_head)
            .ok_or_else(|| anyhow!("d_model overflow"))?;
        let elements = seq_len
            .checked_mul(d_model)
            .ok_or_else(|| anyhow!("attention shape is too large"))?;
        let allocator = Arc::new(memory::GpuAllocator);
        let bytes = elements * std::mem::size_of::<f32>();
        let workspace_bytes = n_heads * seq_len * seq_len * std::mem::size_of::<f32>();
        Ok(Self {
            n_heads,
            seq_len,
            d_head,
            d_model,
            q: memory::CudaBuffer::new(bytes, allocator.clone())?,
            k: memory::CudaBuffer::new(bytes, allocator.clone())?,
            v: memory::CudaBuffer::new(bytes, allocator.clone())?,
            d_out: memory::CudaBuffer::new(bytes, allocator.clone())?,
            out: memory::CudaBuffer::new(bytes, allocator.clone())?,
            d_q: memory::CudaBuffer::new(bytes, allocator.clone())?,
            d_k: memory::CudaBuffer::new(bytes, allocator.clone())?,
            d_v: memory::CudaBuffer::new(bytes, allocator.clone())?,
            probabilities: memory::CudaBuffer::new(workspace_bytes, allocator)?,
        })
    }

    pub fn forward_device(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        scale: f32,
        causal: bool,
    ) -> Result<DeviceTensor> {
        let mut out = DeviceTensor::zeros(vec![self.seq_len, self.d_model])?;
        self.forward_device_into(q, k, v, scale, causal, &mut out)?;
        Ok(out)
    }

    pub fn forward_device_into(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        scale: f32,
        causal: bool,
        out: &mut DeviceTensor,
    ) -> Result<()> {
        if out.shape != vec![self.seq_len, self.d_model] {
            return Err(anyhow!("attention output shape mismatch"));
        }

        if self.n_heads == 1 {
            kernels::cuda_fused_attention(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                out.as_mut_ptr(),
                self.seq_len,
                self.d_head,
                scale,
                causal,
            )?;
        } else {
            kernels::cuda_fused_mha_forward(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                out.as_mut_ptr(),
                self.n_heads,
                self.seq_len,
                self.d_head,
                scale,
                causal,
            )?;
        }
        Ok(())
    }

    pub fn backward_device(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        d_out: &DeviceTensor,
        scale: f32,
        causal: bool,
    ) -> Result<(DeviceTensor, DeviceTensor, DeviceTensor)> {
        let mut d_q = DeviceTensor::zeros(vec![self.seq_len, self.d_model])?;
        let mut d_k = DeviceTensor::zeros(vec![self.seq_len, self.d_model])?;
        let mut d_v = DeviceTensor::zeros(vec![self.seq_len, self.d_model])?;
        self.backward_device_into(q, k, v, d_out, scale, causal, &mut d_q, &mut d_k, &mut d_v)?;
        Ok((d_q, d_k, d_v))
    }

    pub fn backward_device_into(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        d_out: &DeviceTensor,
        scale: f32,
        causal: bool,
        d_q: &mut DeviceTensor,
        d_k: &mut DeviceTensor,
        d_v: &mut DeviceTensor,
    ) -> Result<()> {
        if d_q.shape != vec![self.seq_len, self.d_model]
            || d_k.shape != vec![self.seq_len, self.d_model]
            || d_v.shape != vec![self.seq_len, self.d_model]
        {
            return Err(anyhow!("attention backward output shape mismatch"));
        }
        if self.n_heads == 1 {
            kernels::cuda_attention_backward(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                d_out.as_ptr(),
                d_q.as_mut_ptr(),
                d_k.as_mut_ptr(),
                d_v.as_mut_ptr(),
                self.probabilities.ptr as *mut f32,
                self.seq_len,
                self.d_head,
                scale,
                causal,
            )?;
        } else {
            kernels::cuda_fused_mha_backward(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                d_out.as_ptr(),
                d_q.as_mut_ptr(),
                d_k.as_mut_ptr(),
                d_v.as_mut_ptr(),
                self.probabilities.ptr as *mut f32,
                self.n_heads,
                self.seq_len,
                self.d_head,
                scale,
                causal,
            )?;
        }
        Ok(())
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        scale: f32,
        causal: bool,
    ) -> Result<Vec<f32>> {
        self.q.upload(q)?;
        self.k.upload(k)?;
        self.v.upload(v)?;
        kernels::cuda_fused_attention(
            self.q.ptr as *const f32,
            self.k.ptr as *const f32,
            self.v.ptr as *const f32,
            self.out.ptr as *mut f32,
            self.seq_len,
            self.d_head,
            scale,
            causal,
        )?;
        let mut out = Vec::new();
        self.out.download(&mut out)?;
        Ok(out)
    }

    pub fn backward(
        &self,
        d_out: &[f32],
        scale: f32,
        causal: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.d_out.upload(d_out)?;
        kernels::cuda_attention_backward(
            self.q.ptr as *const f32,
            self.k.ptr as *const f32,
            self.v.ptr as *const f32,
            self.d_out.ptr as *const f32,
            self.d_q.ptr as *mut f32,
            self.d_k.ptr as *mut f32,
            self.d_v.ptr as *mut f32,
            self.probabilities.ptr as *mut f32,
            self.seq_len,
            self.d_head,
            scale,
            causal,
        )?;
        let mut d_q = Vec::new();
        let mut d_k = Vec::new();
        let mut d_v = Vec::new();
        self.d_q.download(&mut d_q)?;
        self.d_k.download(&mut d_k)?;
        self.d_v.download(&mut d_v)?;
        Ok((d_q, d_k, d_v))
    }
}

/// Run fused causal attention from host slices through the CUDA launcher.
pub fn fused_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    d_head: usize,
    scale: f32,
    causal: bool,
) -> Result<Vec<f32>> {
    if d_head == 0 || d_head > 128 {
        return Err(anyhow!("d_head must be in the range 1..=128"));
    }
    let expected = seq_len
        .checked_mul(d_head)
        .ok_or_else(|| anyhow!("attention shape is too large"))?;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(anyhow!(
            "q, k, and v must each have shape [seq_len, d_head]"
        ));
    }

    let allocator = Arc::new(memory::GpuAllocator);
    let q_device =
        memory::CudaBuffer::new(expected * std::mem::size_of::<f32>(), allocator.clone())?;
    let k_device =
        memory::CudaBuffer::new(expected * std::mem::size_of::<f32>(), allocator.clone())?;
    let v_device =
        memory::CudaBuffer::new(expected * std::mem::size_of::<f32>(), allocator.clone())?;
    let out_device = memory::CudaBuffer::new(expected * std::mem::size_of::<f32>(), allocator)?;

    q_device.upload(q)?;
    k_device.upload(k)?;
    v_device.upload(v)?;
    kernels::cuda_fused_attention(
        q_device.ptr as *const f32,
        k_device.ptr as *const f32,
        v_device.ptr as *const f32,
        out_device.ptr as *mut f32,
        seq_len,
        d_head,
        scale,
        causal,
    )?;

    let mut out = Vec::new();
    out_device.download(&mut out)?;
    Ok(out)
}

/// Run the direct unfused attention backward baseline from host slices.
pub fn attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    seq_len: usize,
    d_head: usize,
    scale: f32,
    causal: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if d_head == 0 || d_head > 128 {
        return Err(anyhow!("d_head must be in the range 1..=128"));
    }
    let expected = seq_len
        .checked_mul(d_head)
        .ok_or_else(|| anyhow!("attention shape is too large"))?;
    if q.len() != expected || k.len() != expected || v.len() != expected || d_out.len() != expected
    {
        return Err(anyhow!(
            "q, k, v, and d_out must each have shape [seq_len, d_head]"
        ));
    }
    let allocator = Arc::new(memory::GpuAllocator);
    let bytes = expected * std::mem::size_of::<f32>();
    let workspace_bytes = seq_len * seq_len * std::mem::size_of::<f32>();
    let q_device = memory::CudaBuffer::new(bytes, allocator.clone())?;
    let k_device = memory::CudaBuffer::new(bytes, allocator.clone())?;
    let v_device = memory::CudaBuffer::new(bytes, allocator.clone())?;
    let dout_device = memory::CudaBuffer::new(bytes, allocator.clone())?;
    let dq_device = memory::CudaBuffer::new(bytes, allocator.clone())?;
    let dk_device = memory::CudaBuffer::new(bytes, allocator.clone())?;
    let dv_device = memory::CudaBuffer::new(bytes, allocator.clone())?;
    let probabilities = memory::CudaBuffer::new(workspace_bytes, allocator)?;
    q_device.upload(q)?;
    k_device.upload(k)?;
    v_device.upload(v)?;
    dout_device.upload(d_out)?;
    kernels::cuda_attention_backward(
        q_device.ptr as *const f32,
        k_device.ptr as *const f32,
        v_device.ptr as *const f32,
        dout_device.ptr as *const f32,
        dq_device.ptr as *mut f32,
        dk_device.ptr as *mut f32,
        dv_device.ptr as *mut f32,
        probabilities.ptr as *mut f32,
        seq_len,
        d_head,
        scale,
        causal,
    )?;
    let mut dq = Vec::new();
    let mut dk = Vec::new();
    let mut dv = Vec::new();
    dq_device.download(&mut dq)?;
    dk_device.download(&mut dk)?;
    dv_device.download(&mut dv)?;
    Ok((dq, dk, dv))
}

/// Returns true if CUDA kernels were compiled in.
pub fn cuda_available() -> bool {
    #[cfg(darkforest_cuda_kernels)]
    return true;
    #[cfg(not(darkforest_cuda_kernels))]
    return false;
}

#[cfg(all(test, darkforest_cuda_kernels))]
mod tests {
    use super::*;

    fn reference_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        d_head: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0; seq_len * d_head];
        for row in 0..seq_len {
            let max_key = if causal { row + 1 } else { seq_len };
            let mut scores = Vec::with_capacity(max_key);
            for key in 0..max_key {
                let score = (0..d_head)
                    .map(|d| q[row * d_head + d] * k[key * d_head + d])
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores
                .iter()
                .map(|score| (score - max_score).exp())
                .collect();
            let normalizer: f32 = weights.iter().sum();
            for key in 0..max_key {
                let weight = weights[key] / normalizer;
                for d in 0..d_head {
                    out[row * d_head + d] += weight * v[key * d_head + d];
                }
            }
        }
        out
    }

    fn reference_backward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_out: &[f32],
        seq_len: usize,
        d_head: usize,
        scale: f32,
        causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut d_q = vec![0.0; seq_len * d_head];
        let mut d_k = vec![0.0; seq_len * d_head];
        let mut d_v = vec![0.0; seq_len * d_head];
        for row in 0..seq_len {
            let max_key = if causal { row + 1 } else { seq_len };
            let mut probabilities = vec![0.0; max_key];
            for key in 0..max_key {
                let score = (0..d_head)
                    .map(|d| q[row * d_head + d] * k[key * d_head + d])
                    .sum::<f32>()
                    * scale;
                probabilities[key] = score;
            }
            let max_score = probabilities
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let normalizer: f32 = probabilities
                .iter()
                .map(|score| (score - max_score).exp())
                .sum();
            for probability in &mut probabilities {
                *probability = (*probability - max_score).exp() / normalizer;
            }

            let d_probability: Vec<f32> = (0..max_key)
                .map(|key| {
                    (0..d_head)
                        .map(|d| d_out[row * d_head + d] * v[key * d_head + d])
                        .sum()
                })
                .collect();
            let correction: f32 = probabilities
                .iter()
                .zip(d_probability.iter())
                .map(|(probability, gradient)| probability * gradient)
                .sum();
            for key in 0..max_key {
                let d_score = probabilities[key] * (d_probability[key] - correction);
                for d in 0..d_head {
                    d_q[row * d_head + d] += scale * d_score * k[key * d_head + d];
                    d_k[key * d_head + d] += scale * d_score * q[row * d_head + d];
                    d_v[key * d_head + d] += probabilities[key] * d_out[row * d_head + d];
                }
            }
        }
        (d_q, d_k, d_v)
    }

    fn objective(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_out: &[f32],
        seq_len: usize,
        d_head: usize,
        scale: f32,
    ) -> f32 {
        fused_attention_forward(q, k, v, seq_len, d_head, scale, true)
            .unwrap()
            .iter()
            .zip(d_out.iter())
            .map(|(output, gradient)| output * gradient)
            .sum()
    }

    #[test]
    fn fused_attention_matches_cpu_reference() {
        assert!(cuda_available());
        let seq_len = 5;
        let d_head = 4;
        let q: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 - 7.0) / 11.0)
            .collect();
        let k: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (3.0 - i as f32) / 13.0)
            .collect();
        let v: Vec<f32> = (0..seq_len * d_head).map(|i| (i as f32).sin()).collect();
        let scale = (d_head as f32).sqrt().recip();

        let actual = fused_attention_forward(&q, &k, &v, seq_len, d_head, scale, true).unwrap();
        let expected = reference_attention(&q, &k, &v, seq_len, d_head, scale, true);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 2e-4,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn device_tensor_round_trips_host_data() {
        let input = vec![1.0f32, -2.0, 3.5, 4.25];
        let device_tensor = DeviceTensor::from_host(&input, vec![2, 2]).unwrap();
        assert_eq!(device_tensor.shape, vec![2, 2]);
        assert_eq!(device_tensor.numel(), input.len());
        let output = device_tensor.download().unwrap();
        for (actual, expected) in output.iter().zip(input.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn device_tensor_matmul_matches_cpu_reference() {
        let a = DeviceTensor::from_host(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_host(&[2.0, -1.0, 0.5, 3.0, 4.0, -2.0], vec![3, 2]).unwrap();
        let actual = a.matmul(&b).unwrap().download().unwrap();
        let expected = [15.0, -1.0, 34.5, -1.0];
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5);
        }
    }

    #[test]
    fn device_tensor_softmax_matches_cpu_reference() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let device = DeviceTensor::from_host(&input, vec![2, 3]).unwrap();
        let actual = device.softmax().unwrap().download().unwrap();
        let expected: Vec<f32> = {
            let row0 = &input[..3];
            let row1 = &input[3..];
            let mut out = Vec::new();
            for row in [row0, row1] {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
                let sum: f32 = exp.iter().sum();
                out.extend(exp.iter().map(|v| v / sum));
            }
            out
        };
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn device_tensor_adamw_matches_cpu_reference() {
        let parameter = DeviceTensor::from_host(&[1.0, -2.0, 0.5], vec![3]).unwrap();
        let first_moment = DeviceTensor::zeros(vec![3]).unwrap();
        let second_moment = DeviceTensor::zeros(vec![3]).unwrap();
        let gradient = DeviceTensor::from_host(&[0.2, -0.4, 0.8], vec![3]).unwrap();
        let lr = 1e-2;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let weight_decay = 0.1;
        DeviceTensor::adamw_update(
            &parameter,
            &first_moment,
            &second_moment,
            &gradient,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            1.0 - beta1,
            1.0 - beta2,
        )
        .unwrap();

        let actual = parameter.download().unwrap();
        let mut expected = vec![1.0, -2.0, 0.5];
        for i in 0..expected.len() {
            let decayed = expected[i] * (1.0 - lr * weight_decay);
            let m = (1.0 - beta1) * [0.2, -0.4, 0.8][i];
            let v = (1.0 - beta2) * [0.2f32, -0.4, 0.8][i].powi(2);
            expected[i] = decayed - lr * m / (1.0 - beta1) / ((v / (1.0 - beta2)).sqrt() + eps);
        }
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn persistent_adamw_state_matches_two_cpu_steps() {
        let initial = vec![1.0f32, -2.0, 0.5];
        let gradients = [vec![0.2f32, -0.4, 0.8], vec![-0.1f32, 0.3, 0.2]];
        let mut state = AdamWState::new(&initial, vec![3]).unwrap();
        let (lr, beta1, beta2, eps, weight_decay) = (1e-2, 0.9, 0.999, 1e-8, 0.1);
        let mut expected = initial.clone();
        let mut m = vec![0.0; 3];
        let mut v = vec![0.0; 3];
        for (step, gradient) in gradients.iter().enumerate() {
            state
                .step(gradient, lr, beta1, beta2, eps, weight_decay)
                .unwrap();
            let t = (step + 1) as f32;
            for i in 0..expected.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * gradient[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * gradient[i] * gradient[i];
                expected[i] = expected[i] * (1.0 - lr * weight_decay)
                    - lr * (m[i] / (1.0 - beta1.powf(t)))
                        / ((v[i] / (1.0 - beta2.powf(t))).sqrt() + eps);
            }
        }
        let actual = state.parameter().unwrap();
        assert_eq!(state.step_count(), 2);
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn attention_backward_matches_cpu_and_finite_difference() {
        assert!(cuda_available());
        let seq_len = 4;
        let d_head = 3;
        let q: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 - 4.0) / 9.0)
            .collect();
        let k: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (2.0 - i as f32) / 10.0)
            .collect();
        let v: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 + 1.0) / 8.0)
            .collect();
        let d_out: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 - 3.0) / 7.0)
            .collect();
        let scale = (d_head as f32).sqrt().recip();

        let actual = attention_backward(&q, &k, &v, &d_out, seq_len, d_head, scale, true).unwrap();
        let expected = reference_backward(&q, &k, &v, &d_out, seq_len, d_head, scale, true);
        for (actual, expected) in actual
            .0
            .iter()
            .zip(expected.0.iter())
            .chain(actual.1.iter().zip(expected.1.iter()))
            .chain(actual.2.iter().zip(expected.2.iter()))
        {
            assert!(
                (actual - expected).abs() <= 2e-4,
                "actual={actual}, expected={expected}"
            );
        }

        let delta = 1e-3;
        for (input, analytical) in [(&q, &actual.0), (&k, &actual.1), (&v, &actual.2)] {
            for index in 0..input.len() {
                let mut plus = input.to_vec();
                let mut minus = input.to_vec();
                plus[index] += delta;
                minus[index] -= delta;
                let numerical = match analytical as *const _ {
                    pointer if std::ptr::eq(pointer, &actual.0) => {
                        let mut q_plus = q.clone();
                        q_plus[index] = plus[index];
                        let mut q_minus = q.clone();
                        q_minus[index] = minus[index];
                        (objective(&q_plus, &k, &v, &d_out, seq_len, d_head, scale)
                            - objective(&q_minus, &k, &v, &d_out, seq_len, d_head, scale))
                            / (2.0 * delta)
                    }
                    pointer if std::ptr::eq(pointer, &actual.1) => {
                        let mut k_plus = k.clone();
                        k_plus[index] = plus[index];
                        let mut k_minus = k.clone();
                        k_minus[index] = minus[index];
                        (objective(&q, &k_plus, &v, &d_out, seq_len, d_head, scale)
                            - objective(&q, &k_minus, &v, &d_out, seq_len, d_head, scale))
                            / (2.0 * delta)
                    }
                    _ => {
                        let mut v_plus = v.clone();
                        v_plus[index] = plus[index];
                        let mut v_minus = v.clone();
                        v_minus[index] = minus[index];
                        (objective(&q, &k, &v_plus, &d_out, seq_len, d_head, scale)
                            - objective(&q, &k, &v_minus, &d_out, seq_len, d_head, scale))
                            / (2.0 * delta)
                    }
                };
                assert!(
                    (analytical[index] - numerical).abs() <= 2e-3,
                    "index={index}, analytical={}, numerical={numerical}",
                    analytical[index]
                );
            }
        }
    }

    #[test]
    fn device_tensor_gelu_matches_cpu() {
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let t = DeviceTensor::from_host(&input, vec![5]).unwrap();
        let actual = t.gelu().unwrap().download().unwrap();
        let expected: Vec<f32> = input
            .iter()
            .map(|&v| {
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                let inner = c * (v + 0.044715 * v.powi(3));
                0.5 * v * (1.0 + inner.tanh())
            })
            .collect();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1e-5, "a={a}, e={e}");
        }
    }

    #[test]
    fn device_tensor_embedding_lookup_matches_cpu() {
        let weight_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = DeviceTensor::from_host(&weight_data, vec![3, 3]).unwrap();
        let indices = vec![2, 0, 1, 2];
        let out = DeviceTensor::embedding_lookup(&indices, &weight)
            .unwrap()
            .download()
            .unwrap();
        assert_eq!(
            out,
            vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,]
        );
    }
}
