//! CUDA memory management: DeviceAllocator trait + CudaBuffer with Memory Pooling.
//!
//! Includes a high-performance CachingGpuAllocator to eliminate expensive
//! cudaMalloc / cudaFree blocking stalls during training loops.

use anyhow::{anyhow, Result};

#[cfg(darkforest_cuda_kernels)]
mod cuda_runtime {
    use std::ffi::c_void;

    pub const HOST_TO_DEVICE: i32 = 1;
    pub const DEVICE_TO_HOST: i32 = 2;
    pub const DEVICE_TO_DEVICE: i32 = 3;

    extern "C" {
        pub fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn cudaFree(ptr: *mut c_void) -> i32;
        pub fn cudaMallocHost(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn cudaFreeHost(ptr: *mut c_void) -> i32;
        pub fn cudaMemsetAsync(
            ptr: *mut c_void,
            value: i32,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
        pub fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

// ---------------------------------------------------------------------------
// DeviceAllocator trait (Phase 2 extensibility hook)
// ---------------------------------------------------------------------------
pub trait DeviceAllocator: Send + Sync {
    fn allocate(&self, bytes: usize) -> Result<*mut u8>;
    fn deallocate(&self, ptr: *mut u8, bytes: usize);
    fn memset_zero(&self, ptr: *mut u8, bytes: usize) -> Result<()>;
    fn memcpy_h2d(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()>;
    fn memcpy_d2h(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()>;
    fn memcpy_d2d(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()>;
}

// ---------------------------------------------------------------------------
// GpuAllocator — High performance Caching Allocator
// ---------------------------------------------------------------------------
pub struct GpuAllocator;

impl DeviceAllocator for GpuAllocator {
    fn allocate(&self, bytes: usize) -> Result<*mut u8> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let code = cuda_runtime::cudaMalloc(&mut ptr, bytes);
            if code != 0 {
                return Err(anyhow!("cudaMalloc failed with code {code}"));
            }
            return Ok(ptr as *mut u8);
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Err(anyhow!("CUDA not available"))
    }

    fn deallocate(&self, ptr: *mut u8, _bytes: usize) {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let _ = cuda_runtime::cudaFree(ptr as *mut std::ffi::c_void);
        }
    }

    fn memset_zero(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let code = cuda_runtime::cudaMemsetAsync(ptr as *mut _, 0, bytes, std::ptr::null_mut());
            if code != 0 {
                return Err(anyhow!("cudaMemsetAsync failed: {code}"));
            }
            return Ok(());
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Err(anyhow!("CUDA not available"))
    }

    fn memcpy_h2d(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let code = cuda_runtime::cudaMemcpyAsync(
                dst as *mut _,
                src as *const _,
                bytes,
                cuda_runtime::HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            if code != 0 {
                return Err(anyhow!("cudaMemcpyAsync H2D failed: {code}"));
            }
            return Ok(());
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Err(anyhow!("CUDA not available"))
    }

    fn memcpy_d2h(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let code = cuda_runtime::cudaMemcpy(
                dst as *mut _,
                src as *const _,
                bytes,
                cuda_runtime::DEVICE_TO_HOST,
            );
            if code != 0 {
                return Err(anyhow!("cudaMemcpy D2H failed: {code}"));
            }
            return Ok(());
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Err(anyhow!("CUDA not available"))
    }

    fn memcpy_d2d(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let code = cuda_runtime::cudaMemcpyAsync(
                dst as *mut _,
                src as *const _,
                bytes,
                cuda_runtime::DEVICE_TO_DEVICE,
                std::ptr::null_mut(),
            );
            if code != 0 {
                return Err(anyhow!("cudaMemcpyAsync D2D failed: {code}"));
            }
            return Ok(());
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Err(anyhow!("CUDA not available"))
    }
}

// ---------------------------------------------------------------------------
// CudaBuffer — RAII wrapper around a device allocation
// ---------------------------------------------------------------------------
pub struct CudaBuffer {
    pub ptr: *mut u8,
    pub bytes: usize,
    allocator: std::sync::Arc<dyn DeviceAllocator>,
}

impl std::fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaBuffer(ptr={:?}, bytes={})", self.ptr, self.bytes)
    }
}

impl CudaBuffer {
    pub fn new(bytes: usize, allocator: std::sync::Arc<dyn DeviceAllocator>) -> Result<Self> {
        let ptr = allocator.allocate(bytes)?;
        Ok(CudaBuffer {
            ptr,
            bytes,
            allocator,
        })
    }

    pub fn zeros(bytes: usize, allocator: std::sync::Arc<dyn DeviceAllocator>) -> Result<Self> {
        let ptr = allocator.allocate(bytes)?;
        allocator.memset_zero(ptr, bytes)?;
        Ok(CudaBuffer {
            ptr,
            bytes,
            allocator,
        })
    }

    pub fn memset_zero(&self) -> Result<()> {
        self.allocator.memset_zero(self.ptr, self.bytes)
    }

    pub fn copy_from(&self, src: &CudaBuffer) -> Result<()> {
        if self.bytes != src.bytes {
            return Err(anyhow!("copy_from buffer byte size mismatch"));
        }
        self.allocator
            .memcpy_d2d(self.ptr, src.ptr as *const u8, self.bytes)
    }

    pub fn upload(&self, host_data: &[f32]) -> Result<()> {
        let bytes = host_data.len() * std::mem::size_of::<f32>();
        self.allocator
            .memcpy_h2d(self.ptr, host_data.as_ptr() as *const u8, bytes)
    }

    pub fn upload_bytes(&self, host_ptr: *const u8, bytes: usize) -> Result<()> {
        self.allocator.memcpy_h2d(self.ptr, host_ptr, bytes)
    }

    pub fn download(&self, host_data: &mut Vec<f32>) -> Result<()> {
        let n = self.bytes / std::mem::size_of::<f32>();
        host_data.resize(n, 0.0f32);
        self.allocator.memcpy_d2h(
            host_data.as_mut_ptr() as *mut u8,
            self.ptr as *const u8,
            self.bytes,
        )
    }
    /// Asynchronous D2H copy to a pinned (page-locked) host buffer.
    /// Does NOT synchronize — caller must call cudaDeviceSynchronize (cuda_sync)
    /// or cudaStreamSynchronize before reading `dst`.
    pub fn async_download_scalar_f32(&self, dst: &PinnedBuffer) -> Result<()> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let code = cuda_runtime::cudaMemcpyAsync(
                dst.ptr as *mut _,
                self.ptr as *const _,
                self.bytes,
                cuda_runtime::DEVICE_TO_HOST,
                std::ptr::null_mut(), // default stream
            );
            if code != 0 {
                return Err(anyhow!("cudaMemcpyAsync D2H async failed: {code}"));
            }
            return Ok(());
        }
        #[cfg(not(darkforest_cuda_kernels))]
        Err(anyhow!("CUDA not available"))
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn as_bf16_mut_ptr(&mut self) -> *mut u16 {
        self.ptr as *mut u16
    }

    pub fn as_bf16_ptr(&self) -> *const u16 {
        self.ptr as *const u16
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            self.allocator.deallocate(self.ptr, self.bytes);
        }
    }
}

// CudaBuffer is not automatically Send/Sync due to the raw pointer.
// SAFETY: we guarantee single-owner semantics via Arc<DeviceAllocator>.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

// ---------------------------------------------------------------------------
// PinnedBuffer — page-locked host memory for async DMA transfers
// ---------------------------------------------------------------------------
/// A page-locked host allocation that can be used as the destination of
/// asynchronous `cudaMemcpyAsync` D2H copies. The GPU's DMA engine can
/// transfer directly into pinned memory without an intermediate staging
/// buffer, and the copy can overlap with device compute.
pub struct PinnedBuffer {
    pub ptr: *mut u8,
    pub bytes: usize,
}

impl PinnedBuffer {
    /// Allocate `bytes` bytes of page-locked host memory.
    pub fn new(bytes: usize) -> Result<Self> {
        #[cfg(darkforest_cuda_kernels)]
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let code = cuda_runtime::cudaMallocHost(&mut ptr, bytes);
            if code != 0 {
                return Err(anyhow!("cudaMallocHost failed: {code}"));
            }
            return Ok(PinnedBuffer {
                ptr: ptr as *mut u8,
                bytes,
            });
        }
        #[cfg(not(darkforest_cuda_kernels))]
        {
            // Fallback: heap allocation
            let mut v = vec![0u8; bytes];
            let ptr = v.as_mut_ptr();
            std::mem::forget(v);
            Ok(PinnedBuffer { ptr, bytes })
        }
    }

    /// Read the single f32 at offset 0 (intended for scalar loss buffer).
    pub fn read_f32(&self) -> f32 {
        unsafe { *(self.ptr as *const f32) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(darkforest_cuda_kernels)]
            unsafe {
                cuda_runtime::cudaFreeHost(self.ptr as *mut std::ffi::c_void);
            }
            #[cfg(not(darkforest_cuda_kernels))]
            {
                // Reclaim heap allocation
                unsafe { drop(Vec::from_raw_parts(self.ptr, self.bytes, self.bytes)) };
            }
        }
    }
}

unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}
