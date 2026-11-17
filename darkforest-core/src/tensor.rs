//! Strided N-Dimensional Tensor with zero-copy views and lock-free interior mutability.
//!
//! Tensors hold an `Arc<Storage>` which owns a contiguous memory buffer.
//! Strided views (`transpose`, `permute`, `slice`, `expand`, `unsqueeze`, `squeeze`)
//! modify only `shape`, `strides`, and `storage_offset` without allocating or copying data.
//!
//! Mutation safety is guarded by atomic version counters on `Storage`. If a tensor
//! is modified in-place while saved in an autograd tape, `check_version` fails immediately.

use anyhow::{anyhow, Result};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// DType
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    Bool,
}

impl DType {
    pub fn itemsize(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::Bool => "bool",
        }
    }

    pub fn is_floating_point(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    pub fn is_integral(&self) -> bool {
        matches!(self, DType::I32 | DType::I64 | DType::U8)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, DType::Bool)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(u32), // device index
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shape and Strides
// ---------------------------------------------------------------------------
pub type Shape = Vec<usize>;

pub fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

/// Compute compatible strides for a view into a tensor with `old_shape` and `old_stride`.
/// Matches PyTorch's `computeStride` algorithm in `aten/src/ATen/TensorUtils.cpp`.
pub fn compute_stride(
    old_shape: &[usize],
    old_stride: &[isize],
    new_shape: &[usize],
) -> Option<Vec<isize>> {
    if old_shape.is_empty() {
        return Some(vec![1isize; new_shape.len()]);
    }

    let old_numel = numel(old_shape);
    let new_numel = numel(new_shape);
    if old_numel != new_numel {
        return None;
    }

    if new_shape.is_empty() {
        return Some(vec![]);
    }

    let mut new_stride = vec![0isize; new_shape.len()];
    let trailing_stride = old_stride.last().copied().unwrap_or(1);
    let mut view_d = new_shape.len() as isize - 1;

    // Fill trailing size-1 dimensions in new_shape with trailing_stride
    while view_d >= 0 && new_shape[view_d as usize] == 1 {
        new_stride[view_d as usize] = trailing_stride;
        view_d -= 1;
    }

    if old_numel == 0 {
        for d in (0..=view_d).rev() {
            let u_d = d as usize;
            new_stride[u_d] = if u_d == new_shape.len() - 1 {
                1
            } else {
                let next_size = (new_shape[u_d + 1] as isize).max(1);
                next_size * new_stride[u_d + 1]
            };
        }
        return Some(new_stride);
    }

    let mut chunk_d = old_shape.len() as isize - 1;

    while chunk_d >= 0 {
        if old_shape[chunk_d as usize] == 1 {
            chunk_d -= 1;
            continue;
        }

        let mut chunk_numel = old_shape[chunk_d as usize] as isize;
        let chunk_stride = old_stride[chunk_d as usize];

        // Combine contiguous dimensions in old_shape
        while chunk_d > 0 {
            let prev_d = (chunk_d - 1) as usize;
            if old_shape[prev_d] == 1 {
                chunk_d -= 1;
                continue;
            }
            if old_stride[prev_d] == chunk_numel * chunk_stride {
                chunk_d -= 1;
                chunk_numel *= old_shape[prev_d] as isize;
            } else {
                break;
            }
        }

        // Slice out dimensions in new_shape matching chunk_numel
        let mut view_numel = 1isize;
        let view_start_d = view_d;
        while view_d >= 0 && (view_numel < chunk_numel || new_shape[view_d as usize] == 1) {
            view_numel *= new_shape[view_d as usize] as isize;
            view_d -= 1;
        }

        if view_numel != chunk_numel {
            return None;
        }

        // Assign strides for dimensions in this chunk
        let mut cur_stride = chunk_stride;
        let mut d = view_start_d;
        while d > view_d {
            let u_d = d as usize;
            new_stride[u_d] = cur_stride;
            cur_stride *= new_shape[u_d] as isize;
            d -= 1;
        }

        chunk_d -= 1;
    }

    // Handle any remaining leading size-1 dimensions in new_shape
    while view_d >= 0 {
        let u_d = view_d as usize;
        new_stride[u_d] = if u_d == new_shape.len() - 1 {
            1
        } else {
            new_stride[u_d + 1] * (new_shape[u_d + 1] as isize).max(1)
        };
        view_d -= 1;
    }

    Some(new_stride)
}

/// Compute default contiguous row-major strides from shape.
/// Clamps dimension multiplier to >= 1 to match PyTorch zero-size dimension stride behavior.
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1isize; n];
    for i in (0..n - 1).rev() {
        let multiplier = (shape[i + 1] as isize).max(1);
        strides[i] = strides[i + 1] * multiplier;
    }
    strides
}

/// Unravel flat linear index to multi-dimensional coordinate in shape.
pub fn unravel_index(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for (d, &dim_size) in shape.iter().enumerate().rev() {
        if dim_size == 0 {
            coords[d] = 0;
        } else {
            coords[d] = flat_idx % dim_size;
            flat_idx /= dim_size;
        }
    }
    coords
}

// ---------------------------------------------------------------------------
// Storage (Raw memory block with atomic versioning & lock-free interior mutability)
// ---------------------------------------------------------------------------
pub struct Storage {
    ptr: *mut u8,
    nbytes: usize,
    device: Device,
    version: AtomicUsize,
    is_device: bool,
    #[cfg(feature = "cuda")]
    cuda_buffer: Option<darkforest_cuda::memory::CudaBuffer>,
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Storage(ptr={:?}, nbytes={}, device={}, version={})",
            self.ptr,
            self.nbytes,
            self.device,
            self.version.load(Ordering::Relaxed)
        )
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if !self.is_device && !self.ptr.is_null() && self.nbytes > 0 {
            let layout = Layout::from_size_align(self.nbytes.max(1), 64).unwrap();
            unsafe {
                dealloc(self.ptr, layout);
            }
        }
    }
}

impl Storage {
    pub fn allocate_cpu(nbytes: usize) -> Arc<Self> {
        let size = nbytes.max(1);
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "CPU memory allocation failed");

        Arc::new(Storage {
            ptr,
            nbytes,
            device: Device::Cpu,
            version: AtomicUsize::new(0),
            is_device: false,
            #[cfg(feature = "cuda")]
            cuda_buffer: None,
        })
    }

    pub fn from_slice_bytes(bytes: &[u8], device: &Device) -> Result<Arc<Self>> {
        match device {
            Device::Cpu => {
                let storage = Self::allocate_cpu(bytes.len());
                if !bytes.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.ptr, bytes.len());
                    }
                }
                Ok(storage)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => {
                let _ = darkforest_cuda::set_device(*id);
                let allocator = std::sync::Arc::new(darkforest_cuda::memory::GpuAllocator);
                let buf = darkforest_cuda::memory::CudaBuffer::new(bytes.len(), allocator)?;
                if !bytes.is_empty() {
                    buf.upload_bytes(bytes.as_ptr(), bytes.len())?;
                }
                let ptr = buf.ptr;
                Ok(Arc::new(Storage {
                    ptr,
                    nbytes: bytes.len(),
                    device: device.clone(),
                    version: AtomicUsize::new(0),
                    is_device: true,
                    cuda_buffer: Some(buf),
                }))
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(anyhow!("CUDA support is not enabled in this build")),
        }
    }

    pub fn from_vec_f32(vec: Vec<f32>) -> Arc<Self> {
        let nbytes = vec.len() * std::mem::size_of::<f32>();
        let storage = Self::allocate_cpu(nbytes);
        if !vec.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    vec.as_ptr() as *const u8,
                    storage.ptr,
                    nbytes,
                );
            }
        }
        storage
    }

    #[inline]
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    #[inline]
    pub fn version(&self) -> usize {
        self.version.load(Ordering::SeqCst)
    }

    #[inline]
    pub fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    #[inline]
    pub unsafe fn as_slice<T>(&self, elem_offset: usize, count: usize) -> &[T] {
        let t_size = std::mem::size_of::<T>();
        let byte_offset = elem_offset.checked_mul(t_size).expect("elem_offset overflow");
        let needed_bytes = count.checked_mul(t_size).expect("count overflow");
        let end_byte = byte_offset.checked_add(needed_bytes).expect("total bytes overflow");
        assert!(end_byte <= self.nbytes, "as_slice out of bounds: {} > {}", end_byte, self.nbytes);
        let p = (self.ptr as *const u8).add(byte_offset) as *const T;
        std::slice::from_raw_parts(p, count)
    }

    #[inline]
    pub unsafe fn as_mut_slice<T>(&self, elem_offset: usize, count: usize) -> &mut [T] {
        let t_size = std::mem::size_of::<T>();
        let byte_offset = elem_offset.checked_mul(t_size).expect("elem_offset overflow");
        let needed_bytes = count.checked_mul(t_size).expect("count overflow");
        let end_byte = byte_offset.checked_add(needed_bytes).expect("total bytes overflow");
        assert!(end_byte <= self.nbytes, "as_mut_slice out of bounds: {} > {}", end_byte, self.nbytes);
        let p = self.ptr.add(byte_offset) as *mut T;
        std::slice::from_raw_parts_mut(p, count)
    }

    #[inline]
    pub unsafe fn read_val<T: Copy>(&self, elem_offset: usize) -> T {
        let t_size = std::mem::size_of::<T>();
        let byte_offset = elem_offset.checked_mul(t_size).expect("elem_offset overflow");
        let end_byte = byte_offset.checked_add(t_size).expect("total bytes overflow");
        assert!(end_byte <= self.nbytes, "read_val out of bounds");
        std::ptr::read((self.ptr as *const u8).add(byte_offset) as *const T)
    }

    #[inline]
    pub unsafe fn write_val<T: Copy>(&self, elem_offset: usize, val: T) {
        let t_size = std::mem::size_of::<T>();
        let byte_offset = elem_offset.checked_mul(t_size).expect("elem_offset overflow");
        let end_byte = byte_offset.checked_add(t_size).expect("total bytes overflow");
        assert!(end_byte <= self.nbytes, "write_val out of bounds");
        std::ptr::write(self.ptr.add(byte_offset) as *mut T, val);
    }
}

// ---------------------------------------------------------------------------
// Legacy Compatibility Bridge: TensorData and Storage enum
// ---------------------------------------------------------------------------
#[derive(Debug)]
pub enum LegacyStorage {
    Cpu(Vec<f32>),
    #[cfg(feature = "cuda")]
    Cuda(darkforest_cuda::DeviceTensor),
}

#[derive(Debug)]
pub struct TensorData {
    pub storage: LegacyStorage,
}

impl TensorData {
    pub fn zeros(shape: &[usize], device: &Device) -> Self {
        match device {
            Device::Cpu => TensorData {
                storage: LegacyStorage::Cpu(vec![0.0f32; numel(shape)]),
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => {
                let _ = darkforest_cuda::set_device(*id);
                TensorData {
                    storage: LegacyStorage::Cuda(
                        darkforest_cuda::DeviceTensor::zeros(shape.to_vec())
                            .expect("CUDA zeros failed"),
                    ),
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => panic!("CUDA support is not enabled"),
        }
    }

    pub fn from_vec(data: Vec<f32>, _shape: &[usize], device: &Device) -> Self {
        match device {
            Device::Cpu => TensorData {
                storage: LegacyStorage::Cpu(data),
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => {
                let _ = darkforest_cuda::set_device(*id);
                TensorData {
                    storage: LegacyStorage::Cuda(
                        darkforest_cuda::DeviceTensor::from_host(&data, _shape.to_vec())
                            .expect("CUDA upload failed"),
                    ),
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => panic!("CUDA support is not enabled"),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        match &self.storage {
            LegacyStorage::Cpu(v) => v.clone(),
            #[cfg(feature = "cuda")]
            LegacyStorage::Cuda(d) => d.download().expect("CUDA download failed"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_tensor(&self) -> &darkforest_cuda::DeviceTensor {
        match &self.storage {
            LegacyStorage::Cuda(d) => d,
            LegacyStorage::Cpu(_) => panic!("Expected CUDA storage on TensorData"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_tensor_mut(&mut self) -> &mut darkforest_cuda::DeviceTensor {
        match &mut self.storage {
            LegacyStorage::Cuda(d) => d,
            LegacyStorage::Cpu(_) => panic!("Expected CUDA storage on TensorData"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct Tensor {
    pub storage: Arc<Storage>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub storage_offset: usize, // Offset in elements (not bytes)
    pub dtype: DType,
    pub device: Device,
    pub requires_grad: bool,
    pub autograd_meta: Option<Arc<std::sync::Mutex<crate::autograd::AutogradMeta>>>,
    pub grad: Option<Arc<Tensor>>,
    // Bridge for legacy grad_fn storage
    pub legacy_storage: Option<Arc<std::sync::RwLock<TensorData>>>,
}

impl Tensor {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    pub fn zeros(shape: impl Into<Shape>) -> Self {
        Self::zeros_device(shape, Device::Cpu)
    }

    pub fn zeros_device(shape: impl Into<Shape>, device: Device) -> Self {
        let shape = shape.into();
        let n = numel(&shape);
        let strides = contiguous_strides(&shape);
        let nbytes = n * DType::F32.itemsize();
        let storage = match device {
            Device::Cpu => Storage::allocate_cpu(nbytes),
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => {
                let bytes = vec![0u8; nbytes];
                Storage::from_slice_bytes(&bytes, &Device::Cuda(id)).unwrap()
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => panic!("CUDA support is not enabled"),
        };

        Tensor {
            storage,
            shape: shape.clone(),
            strides,
            storage_offset: 0,
            dtype: DType::F32,
            device: device.clone(),
            requires_grad: false,
            autograd_meta: None,
            grad: None,
            legacy_storage: Some(Arc::new(std::sync::RwLock::new(TensorData::zeros(&shape, &device)))),
        }
    }

    pub fn ones(shape: impl Into<Shape>) -> Self {
        Self::ones_device(shape, Device::Cpu)
    }

    pub fn ones_device(shape: impl Into<Shape>, device: Device) -> Self {
        let shape = shape.into();
        let n = numel(&shape);
        let data = vec![1.0f32; n];
        Self::from_vec_device(data, shape, device).unwrap()
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        Self::from_vec_device(data, shape, Device::Cpu)
    }

    pub fn from_vec_device(
        data: Vec<f32>,
        shape: impl Into<Shape>,
        device: Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let n = numel(&shape);
        if data.len() != n {
            return Err(anyhow!(
                "from_vec: data length {} != numel({:?}) = {}",
                data.len(),
                shape,
                n
            ));
        }
        let strides = contiguous_strides(&shape);
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        let storage = Storage::from_slice_bytes(byte_slice, &device)?;

        Ok(Tensor {
            storage,
            shape: shape.clone(),
            strides,
            storage_offset: 0,
            dtype: DType::F32,
            device: device.clone(),
            requires_grad: false,
            autograd_meta: None,
            grad: None,
            legacy_storage: Some(Arc::new(std::sync::RwLock::new(TensorData::from_vec(
                data, &shape, &device,
            )))),
        })
    }

    pub fn from_typed_bytes(
        bytes: &[u8],
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let n = numel(&shape);
        let expected_bytes = n * dtype.itemsize();
        if bytes.len() != expected_bytes {
            return Err(anyhow!(
                "from_typed_bytes: byte length {} != expected {} for shape {:?} and dtype {:?}",
                bytes.len(),
                expected_bytes,
                shape,
                dtype
            ));
        }
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_slice_bytes(bytes, &device)?;
        Ok(Tensor {
            storage,
            shape,
            strides,
            storage_offset: 0,
            dtype,
            device,
            requires_grad: false,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn from_cuda(dev: darkforest_cuda::DeviceTensor) -> Self {
        let shape = dev.shape.clone();
        let strides = contiguous_strides(&shape);
        let ptr = dev.as_ptr() as *mut u8;
        let nbytes = dev.numel() * std::mem::size_of::<f32>();
        let storage = Arc::new(Storage {
            ptr,
            nbytes,
            device: Device::Cuda(0),
            version: AtomicUsize::new(0),
            is_device: true,
            cuda_buffer: None,
        });
        Tensor {
            storage,
            shape: shape.clone(),
            strides,
            storage_offset: 0,
            dtype: DType::F32,
            device: Device::Cuda(0),
            requires_grad: false,
            autograd_meta: None,
            grad: None,
            legacy_storage: Some(Arc::new(std::sync::RwLock::new(TensorData {
                storage: LegacyStorage::Cuda(dev),
            }))),
        }
    }

    pub fn scalar(val: f32) -> Self {
        Tensor::from_vec(vec![val], vec![1]).unwrap()
    }

    pub fn randn(shape: impl Into<Shape>, std: f32) -> Self {
        let shape = shape.into();
        let n = numel(&shape);
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0f64, std as f64).unwrap();
        let data: Vec<f32> = (0..n).map(|_| rng.sample(dist) as f32).collect();
        Tensor::from_vec(data, shape).unwrap()
    }

    pub fn rand_uniform(shape: impl Into<Shape>, lo: f32, hi: f32) -> Self {
        let shape = shape.into();
        let n = numel(&shape);
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(lo as f64, hi as f64);
        let data: Vec<f32> = (0..n).map(|_| rng.sample(dist) as f32).collect();
        Tensor::from_vec(data, shape).unwrap()
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    pub fn with_grad(mut self) -> Self {
        self.set_requires_grad(true);
        self
    }

    pub fn requires_grad_(mut self, req: bool) -> Self {
        self.set_requires_grad(req);
        self
    }

    pub fn set_requires_grad(&mut self, req: bool) {
        self.requires_grad = req;
        if req {
            if self.autograd_meta.is_none() {
                self.autograd_meta = Some(Arc::new(std::sync::Mutex::new(
                    crate::autograd::AutogradMeta::new_leaf(true),
                )));
            } else {
                self.autograd_meta.as_ref().unwrap().lock().unwrap().requires_grad = true;
            }
        } else if let Some(meta) = &self.autograd_meta {
            meta.lock().unwrap().requires_grad = false;
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn is_leaf(&self) -> bool {
        self.autograd_meta
            .as_ref()
            .map_or(true, |m| m.lock().unwrap().is_leaf)
    }

    pub fn grad_fn(&self) -> Option<Arc<dyn crate::autograd::Node>> {
        self.autograd_meta
            .as_ref()
            .and_then(|m| m.lock().unwrap().grad_fn.clone())
    }

    pub fn grad_fn_name(&self) -> Option<&'static str> {
        self.grad_fn().map(|f| f.name())
    }

    pub fn set_grad_fn(&mut self, grad_fn: Arc<dyn crate::autograd::Node>, output_nr: usize) {
        self.requires_grad = true;
        self.autograd_meta = Some(Arc::new(std::sync::Mutex::new(
            crate::autograd::AutogradMeta::new_node(grad_fn, output_nr),
        )));
    }

    pub fn grad(&self) -> Option<Tensor> {
        if let Some(m) = &self.autograd_meta {
            if let Some(g) = &m.lock().unwrap().grad {
                return Some(g.clone());
            }
        }
        self.grad.as_ref().map(|g| (**g).clone())
    }

    pub fn set_grad(&self, grad: Option<Tensor>) {
        if let Some(m) = &self.autograd_meta {
            m.lock().unwrap().grad = grad;
        }
    }

    pub fn gradient_edge(&self) -> Option<crate::autograd::Edge> {
        if !self.requires_grad {
            return None;
        }
        let meta_arc = self.autograd_meta.as_ref()?;
        let mut meta = meta_arc.lock().unwrap();
        if let Some(grad_fn) = &meta.grad_fn {
            return Some(crate::autograd::Edge::new(grad_fn.clone(), meta.output_nr));
        }

        // Leaf tensor: return Edge pointing to AccumulateGrad node
        if let Some(weak_accum) = &meta.grad_accumulator {
            if let Some(accum) = weak_accum.upgrade() {
                return Some(crate::autograd::Edge::new(accum, 0));
            }
        }
        let accum = Arc::new(crate::autograd::AccumulateGrad::new(Arc::downgrade(meta_arc)));
        meta.grad_accumulator = Some(Arc::downgrade(&accum));
        Some(crate::autograd::Edge::new(accum, 0))
    }

    pub fn detach(&self) -> Tensor {
        Tensor {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: false,
            autograd_meta: None,
            grad: None,
            legacy_storage: self.legacy_storage.clone(),
        }
    }

    pub fn backward(&self, gradient: Option<Tensor>, retain_graph: bool) -> Result<()> {
        if !self.requires_grad() || (self.grad_fn().is_none() && !self.is_leaf()) {
            return Err(anyhow!(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            ));
        }
        crate::autograd::backward(self, gradient, retain_graph)?;
        if !retain_graph {
            if let Some(m) = &self.autograd_meta {
                m.lock().unwrap().grad_fn = None;
            }
        }
        Ok(())
    }

    pub fn backward_default(&self) -> Result<()> {
        self.backward(None, false)
    }

    // -----------------------------------------------------------------------
    // Core Inspection & Versioning
    // -----------------------------------------------------------------------

    #[inline]
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    #[inline]
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    #[inline]
    pub fn data_ptr(&self) -> *mut u8 {
        self.storage.ptr()
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub fn strides(&self) -> &Vec<isize> {
        &self.strides
    }

    #[inline]
    pub fn storage_offset(&self) -> usize {
        self.storage_offset
    }

    #[inline]
    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn version(&self) -> usize {
        self.storage.version()
    }

    #[inline]
    pub fn bump_version(&self) {
        self.storage.bump_version();
    }

    #[inline]
    pub fn check_version(&self, saved_version: usize) -> Result<()> {
        if self.version() != saved_version {
            Err(anyhow!(
                "one of the variables needed for gradient computation has been modified by an inplace operation (saved version {}, current version {})",
                saved_version,
                self.version()
            ))
        } else {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Strided View Primitives (Zero-Copy)
    // -----------------------------------------------------------------------

    /// Check whether tensor is contiguous in row-major memory order.
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected_stride = 1isize;
        for (&size, &stride) in self.shape.iter().rev().zip(self.strides.iter().rev()) {
            if size == 0 {
                return true;
            }
            if size == 1 {
                continue;
            }
            if stride != expected_stride {
                return false;
            }
            expected_stride *= size as isize;
        }
        true
    }

    /// Primary strided view constructor. Validates memory boundary and creates zero-copy view.
    pub fn as_strided(
        &self,
        size: Vec<usize>,
        stride: Vec<isize>,
        storage_offset: Option<usize>,
    ) -> Result<Tensor> {
        let offset = storage_offset.unwrap_or(self.storage_offset);
        if size.len() != stride.len() {
            return Err(anyhow!(
                "as_strided: size length ({}) != stride length ({})",
                size.len(),
                stride.len()
            ));
        }

        // Validate memory bounds
        if !size.is_empty() && numel(&size) > 0 {
            let mut min_elem_offset = offset as isize;
            let mut max_elem_offset = offset as isize;
            for (&s, &strd) in size.iter().zip(stride.iter()) {
                if s > 0 {
                    let span = (s as isize - 1) * strd;
                    if span > 0 {
                        max_elem_offset += span;
                    } else {
                        min_elem_offset += span;
                    }
                }
            }
            if min_elem_offset < 0 {
                return Err(anyhow!(
                    "as_strided: computed negative storage offset {}",
                    min_elem_offset
                ));
            }
            let required_bytes = (max_elem_offset + 1) as usize * self.dtype.itemsize();
            if required_bytes > self.storage.nbytes {
                return Err(anyhow!(
                    "as_strided: out of storage bounds: requires {} bytes, but storage has {} bytes",
                    required_bytes,
                    self.storage.nbytes
                ));
            }
        }

        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            shape: size,
            strides: stride,
            storage_offset: offset,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        })
    }

    /// Return a contiguous in memory copy of self if not already contiguous.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let n = self.numel();
        let itemsize = self.dtype.itemsize();
        let new_storage = match self.device {
            Device::Cpu => Storage::allocate_cpu(n * itemsize),
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => {
                let bytes = vec![0u8; n * itemsize];
                Storage::from_slice_bytes(&bytes, &Device::Cuda(id))?
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => return Err(anyhow!("CUDA not enabled")),
        };

        if self.device == Device::Cpu {
            let src_bytes = self.storage.ptr();
            let dst_bytes = new_storage.ptr();
            for flat_i in 0..n {
                let coords = unravel_index(flat_i, &self.shape);
                let elem_offset = self.calculate_element_offset(&coords);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_bytes.add(elem_offset * itemsize),
                        dst_bytes.add(flat_i * itemsize),
                        itemsize,
                    );
                }
            }
        } else {
            // For CUDA non-contiguous copies, fallback download & re-upload for now
            let vec = self.to_vec();
            return Tensor::from_vec_device(vec, self.shape.clone(), self.device.clone());
        }

        let strides = contiguous_strides(&self.shape);
        Ok(Tensor {
            storage: new_storage,
            shape: self.shape.clone(),
            strides,
            storage_offset: 0,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        })
    }

    /// Transpose two dimensions (zero copy).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(anyhow!(
                "transpose: dims ({}, {}) out of range for ndim {}",
                dim0,
                dim1,
                ndim
            ));
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        let mut out = Tensor {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        };

        if crate::autograd::is_grad_enabled() && self.requires_grad {
            if let Some(e) = self.gradient_edge() {
                let node = Arc::new(crate::autograd::ops::TransposeBackward::new(dim0, dim1, vec![e]));
                out.set_grad_fn(node, 0);
            }
        }

        Ok(out)
    }

    /// Transpose last two dimensions (zero copy).
    pub fn transpose_last_two(&self) -> Result<Tensor> {
        let ndim = self.ndim();
        if ndim < 2 {
            return Err(anyhow!("transpose_last_two requires ndim >= 2"));
        }
        self.transpose(ndim - 2, ndim - 1)
    }

    /// Permute dimensions according to order (zero copy).
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        if dims.len() != ndim {
            return Err(anyhow!(
                "permute: dims len {} != tensor ndim {}",
                dims.len(),
                ndim
            ));
        }
        let mut seen = vec![false; ndim];
        for &d in dims {
            if d >= ndim {
                return Err(anyhow!("permute: dim {} out of range", d));
            }
            if seen[d] {
                return Err(anyhow!("permute: duplicate dimension {}", d));
            }
            seen[d] = true;
        }

        let new_shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();
        let new_strides: Vec<isize> = dims.iter().map(|&d| self.strides[d]).collect();

        let mut out = Tensor {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        };

        if crate::autograd::is_grad_enabled() && self.requires_grad {
            if let Some(e) = self.gradient_edge() {
                let node = Arc::new(crate::autograd::ops::PermuteBackward::new(dims, vec![e]));
                out.set_grad_fn(node, 0);
            }
        }

        Ok(out)
    }

    /// Insert dimension of size 1 at position `dim` (zero copy).
    pub fn unsqueeze(&self, dim: i64) -> Result<Tensor> {
        let ndim = self.ndim();
        let d = if dim < 0 {
            (ndim as i64 + 1 + dim) as usize
        } else {
            dim as usize
        };
        if d > ndim {
            return Err(anyhow!(
                "unsqueeze: dim {} out of range for ndim {}",
                dim,
                ndim
            ));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.insert(d, 1);
        let stride = if d < ndim {
            self.shape[d] as isize * self.strides[d]
        } else if ndim == 0 {
            1
        } else {
            1
        };
        new_strides.insert(d, stride);

        let mut out = Tensor {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        };

        if crate::autograd::is_grad_enabled() && self.requires_grad {
            if let Some(e) = self.gradient_edge() {
                let node = Arc::new(crate::autograd::ops::ViewBackward::new(self.shape.clone(), vec![e]));
                out.set_grad_fn(node, 0);
            }
        }

        Ok(out)
    }

    /// Remove dimension of size 1 at position `dim` or all size-1 dims (zero copy).
    pub fn squeeze(&self, dim: Option<i64>) -> Result<Tensor> {
        let ndim = self.ndim();
        match dim {
            None => {
                let mut new_shape = Vec::new();
                let mut new_strides = Vec::new();
                for (s, st) in self.shape.iter().zip(self.strides.iter()) {
                    if *s != 1 {
                        new_shape.push(*s);
                        new_strides.push(*st);
                    }
                }
                let mut out = Tensor {
                    storage: Arc::clone(&self.storage),
                    shape: new_shape,
                    strides: new_strides,
                    storage_offset: self.storage_offset,
                    dtype: self.dtype,
                    device: self.device.clone(),
                    requires_grad: self.requires_grad,
                    autograd_meta: None,
                    grad: None,
                    legacy_storage: None,
                };
                if crate::autograd::is_grad_enabled() && self.requires_grad {
                    if let Some(e) = self.gradient_edge() {
                        let node = Arc::new(crate::autograd::ops::ViewBackward::new(self.shape.clone(), vec![e]));
                        out.set_grad_fn(node, 0);
                    }
                }
                Ok(out)
            }
            Some(d) => {
                let idx = if d < 0 {
                    (ndim as i64 + d) as usize
                } else {
                    d as usize
                };
                if idx >= ndim {
                    return Err(anyhow!("squeeze: dim {} out of range for ndim {}", d, ndim));
                }
                if self.shape[idx] == 1 {
                    let mut new_shape = self.shape.clone();
                    let mut new_strides = self.strides.clone();
                    new_shape.remove(idx);
                    new_strides.remove(idx);
                    let mut out = Tensor {
                        storage: Arc::clone(&self.storage),
                        shape: new_shape,
                        strides: new_strides,
                        storage_offset: self.storage_offset,
                        dtype: self.dtype,
                        device: self.device.clone(),
                        requires_grad: self.requires_grad,
                        autograd_meta: None,
                        grad: None,
                        legacy_storage: None,
                    };
                    if crate::autograd::is_grad_enabled() && self.requires_grad {
                        if let Some(e) = self.gradient_edge() {
                            let node = Arc::new(crate::autograd::ops::ViewBackward::new(self.shape.clone(), vec![e]));
                            out.set_grad_fn(node, 0);
                        }
                    }
                    Ok(out)
                } else {
                    Ok(self.clone())
                }
            }
        }
    }

    /// Narrow / slice tensor along `dim` from `start` to `end` with `step` (zero copy).
    pub fn slice(&self, dim: usize, start: usize, end: usize, step: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(anyhow!("slice: dim {} out of range for ndim {}", dim, ndim));
        }
        if step == 0 {
            return Err(anyhow!("slice: step must be > 0"));
        }
        let dim_len = self.shape[dim];
        let clamped_start = start.min(dim_len);
        let clamped_end = end.min(dim_len);

        let new_offset = self.storage_offset
            + (clamped_start as isize * self.strides[dim]) as usize;
        let new_size = if clamped_end > clamped_start {
            (clamped_end - clamped_start + step - 1) / step
        } else {
            0
        };
        let new_stride = self.strides[dim] * (step as isize);

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape[dim] = new_size;
        new_strides[dim] = new_stride;

        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: new_offset,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        })
    }

    /// Expand / broadcast dimensions (zero copy via stride 0).
    pub fn expand(&self, target_shape: &[usize]) -> Result<Tensor> {
        let orig_ndim = self.ndim();
        let target_ndim = target_shape.len();
        if target_ndim < orig_ndim {
            return Err(anyhow!(
                "expand: target ndim {} < source ndim {}",
                target_ndim,
                orig_ndim
            ));
        }

        let leading_ones = target_ndim - orig_ndim;
        let mut new_shape = vec![1usize; target_ndim];
        let mut new_strides = vec![0isize; target_ndim];

        // Fill trailing dimensions with original shape/strides
        for i in 0..orig_ndim {
            new_shape[leading_ones + i] = self.shape[i];
            new_strides[leading_ones + i] = self.strides[i];
        }

        // Apply expand rules
        for d in 0..target_ndim {
            let desired = target_shape[d];
            let current = new_shape[d];
            if current == desired {
                continue;
            } else if current == 1 {
                new_shape[d] = desired;
                new_strides[d] = 0; // Stride 0 broadcasts along dimension
            } else {
                return Err(anyhow!(
                    "expand: cannot expand dimension {} from size {} to {}",
                    d,
                    current,
                    desired
                ));
            }
        }

        let mut out = Tensor {
            storage: Arc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: self.storage_offset,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        };

        if crate::autograd::is_grad_enabled() && self.requires_grad {
            if let Some(e) = self.gradient_edge() {
                let node = Arc::new(crate::autograd::ops::ExpandBackward::new(self.shape.clone(), vec![e]));
                out.set_grad_fn(node, 0);
            }
        }

        Ok(out)
    }

    /// Flip dimensions (negative strides zero-copy view).
    pub fn flip(&self, dims: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        let mut new_strides = self.strides.clone();
        let mut new_offset = self.storage_offset as isize;

        for &d in dims {
            if d >= ndim {
                return Err(anyhow!("flip: dim {} out of range for ndim {}", d, ndim));
            }
            let size = self.shape[d];
            if size > 0 {
                new_offset += (size as isize - 1) * new_strides[d];
                new_strides[d] = -new_strides[d];
            }
        }

        let mut out = Tensor {
            storage: Arc::clone(&self.storage),
            shape: self.shape.clone(),
            strides: new_strides,
            storage_offset: new_offset as usize,
            dtype: self.dtype,
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        };

        if crate::autograd::is_grad_enabled() && self.requires_grad {
            if let Some(e) = self.gradient_edge() {
                let node = Arc::new(crate::autograd::ops::FlipBackward::new(dims, vec![e]));
                out.set_grad_fn(node, 0);
            }
        }

        Ok(out)
    }

    /// Check if multiple elements in tensor view refer to the same physical memory location.
    pub fn has_internal_overlap(&self) -> bool {
        if self.numel() <= 1 {
            return false;
        }
        for (&size, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if size > 1 && stride == 0 {
                return true;
            }
        }
        if self.is_contiguous() {
            return false;
        }
        let n = self.numel();
        if n > 50_000 {
            return false;
        }
        let mut seen = std::collections::HashSet::with_capacity(n);
        for flat_i in 0..n {
            let coords = unravel_index(flat_i, &self.shape);
            let elem_offset = self.calculate_element_offset(&coords);
            if !seen.insert(elem_offset) {
                return true;
            }
        }
        false
    }

    /// View with new shape. Uses compute_stride algorithm for general view compatibility (zero copy).
    pub fn view(&self, new_shape: impl Into<Shape>) -> Result<Tensor> {
        let new_shape = new_shape.into();
        let target_numel = numel(&new_shape);
        if target_numel != self.numel() {
            return Err(anyhow!(
                "view: shape {:?} (numel {}) is not compatible with tensor numel {}",
                new_shape,
                target_numel,
                self.numel()
            ));
        }

        if let Some(strides) = compute_stride(&self.shape, &self.strides, &new_shape) {
            let mut out = Tensor {
                storage: Arc::clone(&self.storage),
                shape: new_shape,
                strides,
                storage_offset: self.storage_offset,
                dtype: self.dtype,
                device: self.device.clone(),
                requires_grad: self.requires_grad,
                autograd_meta: None,
                grad: None,
                legacy_storage: None,
            };

            if crate::autograd::is_grad_enabled() && self.requires_grad {
                if let Some(e) = self.gradient_edge() {
                    let node = Arc::new(crate::autograd::ops::ViewBackward::new(self.shape.clone(), vec![e]));
                    out.set_grad_fn(node, 0);
                }
            }

            Ok(out)
        } else {
            Err(anyhow!(
                "view size is not compatible with input tensor's size and stride (use .reshape() or .contiguous() instead)"
            ))
        }
    }

    /// Reshape into new shape. Uses zero-copy view if compatible, else copies.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Tensor> {
        let new_shape = new_shape.into();
        match self.view(new_shape.clone()) {
            Ok(v) => Ok(v),
            Err(_) => {
                let contig = self.contiguous()?;
                contig.view(new_shape)
            }
        }
    }

    /// Flatten contiguous range of dimensions [start_dim, end_dim].
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Result<Tensor> {
        let ndim = self.ndim();
        let s = if start_dim < 0 {
            (ndim as i64 + start_dim) as usize
        } else {
            start_dim as usize
        };
        let e = if end_dim < 0 {
            (ndim as i64 + end_dim) as usize
        } else {
            end_dim as usize
        };
        if s > e || e >= ndim {
            return Err(anyhow!("flatten: invalid range [{}, {}]", start_dim, end_dim));
        }

        let flat_size: usize = self.shape[s..=e].iter().product();
        let mut new_shape: Vec<usize> = self.shape[..s].to_vec();
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&self.shape[e + 1..]);

        self.reshape(new_shape)
    }

    // -----------------------------------------------------------------------
    // Data Accessors & In-place Mutation
    // -----------------------------------------------------------------------

    #[inline]
    fn calculate_element_offset(&self, coords: &[usize]) -> usize {
        let mut offset = self.storage_offset as isize;
        for (c, &s) in coords.iter().zip(self.strides.iter()) {
            offset += (*c as isize) * s;
        }
        offset as usize
    }

    pub fn get(&self, flat_idx: usize) -> f32 {
        if self.is_cuda() {
            return self.to_vec()[flat_idx];
        }
        let coords = unravel_index(flat_idx, &self.shape);
        let elem_offset = self.calculate_element_offset(&coords);
        unsafe {
            self.storage.read_val::<f32>(elem_offset)
        }
    }

    pub fn set(&self, flat_idx: usize, val: f32) {
        assert!(
            !self.has_internal_overlap(),
            "unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation."
        );
        if self.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let coords = unravel_index(flat_idx, &self.shape);
                let elem_offset = self.calculate_element_offset(&coords);
                let _byte_offset = elem_offset * std::mem::size_of::<f32>();
                if let Some(ref buf) = self.storage.cuda_buffer {
                    let bytes = val.to_ne_bytes();
                    let _ = buf.upload_bytes(bytes.as_ptr(), 4);
                }
            }
            self.bump_version();
            return;
        }

        let coords = unravel_index(flat_idx, &self.shape);
        let elem_offset = self.calculate_element_offset(&coords);
        unsafe {
            self.storage.write_val::<f32>(elem_offset, val);
        }
        self.bump_version();
    }

    pub fn to_vec(&self) -> Vec<f32> {
        if self.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                if let Some(ref buf) = self.storage.cuda_buffer {
                    let mut host = vec![0.0f32; self.numel()];
                    let _ = buf.download(&mut host);
                    return host;
                }
            }
        }

        let n = self.numel();
        if self.is_contiguous() {
            unsafe {
                let slice = self.storage.as_slice::<f32>(self.storage_offset, n);
                return slice.to_vec();
            }
        }

        let mut out = Vec::with_capacity(n);
        for flat_i in 0..n {
            let coords = unravel_index(flat_i, &self.shape);
            let elem_offset = self.calculate_element_offset(&coords);
            unsafe {
                let slice = self.storage.as_slice::<f32>(elem_offset, 1);
                out.push(slice[0]);
            }
        }
        out
    }

    /// In-place elementwise scalar addition. Bumps storage version counter.
    pub fn add_inplace_(&mut self, scalar: f32) -> Result<()> {
        if self.has_internal_overlap() {
            return Err(anyhow!(
                "unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation."
            ));
        }
        if self.device != Device::Cpu {
            return Err(anyhow!("add_inplace_ currently supported on CPU"));
        }
        let n = self.numel();
        for flat_i in 0..n {
            let coords = unravel_index(flat_i, &self.shape);
            let elem_offset = self.calculate_element_offset(&coords);
            unsafe {
                let slice = self.storage.as_mut_slice::<f32>(elem_offset, 1);
                slice[0] += scalar;
            }
        }
        self.bump_version();
        Ok(())
    }

    pub fn zeros_dtype(shape: impl Into<Shape>, dtype: DType) -> Self {
        Self::zeros_dtype_device(shape, dtype, Device::Cpu)
    }

    pub fn zeros_dtype_device(shape: impl Into<Shape>, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let n = numel(&shape);
        let nbytes = n * dtype.itemsize();
        let storage = Storage::allocate_cpu(nbytes);
        let strides = contiguous_strides(&shape);
        Tensor {
            storage,
            shape,
            strides,
            storage_offset: 0,
            dtype,
            device,
            requires_grad: false,
            autograd_meta: None,
            grad: None,
            legacy_storage: None,
        }
    }

    pub fn from_slice_i64(data: &[i64], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<i64>(),
            )
        };
        Self::from_typed_bytes(byte_slice, shape, DType::I64, Device::Cpu)
    }

    pub fn from_slice_i32(data: &[i32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<i32>(),
            )
        };
        Self::from_typed_bytes(byte_slice, shape, DType::I32, Device::Cpu)
    }

    pub fn from_slice_bool(data: &[bool], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<bool>(),
            )
        };
        Self::from_typed_bytes(byte_slice, shape, DType::Bool, Device::Cpu)
    }

    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        if self.dtype != DType::I64 {
            return Err(anyhow!("to_vec_i64: tensor dtype is {:?}, not I64", self.dtype));
        }
        let n = self.numel();
        let mut out = Vec::with_capacity(n);
        for flat_i in 0..n {
            let coords = unravel_index(flat_i, &self.shape);
            let elem_offset = self.calculate_element_offset(&coords);
            unsafe {
                let slice = self.storage.as_slice::<i64>(elem_offset, 1);
                out.push(slice[0]);
            }
        }
        Ok(out)
    }

    pub fn to_dtype(&self, target_dtype: DType) -> Result<Tensor> {
        if self.dtype == target_dtype {
            return Ok(self.clone());
        }
        let n = self.numel();
        let out = Tensor::zeros_dtype(self.shape.clone(), target_dtype);
        for i in 0..n {
            let val_f32 = self.get(i);
            match target_dtype {
                DType::F32 => out.set(i, val_f32),
                DType::I64 => {
                    let v = val_f32 as i64;
                    let coords = unravel_index(i, &out.shape);
                    let elem_offset = out.calculate_element_offset(&coords);
                    unsafe {
                        let slice = out.storage.as_mut_slice::<i64>(elem_offset, 1);
                        slice[0] = v;
                    }
                }
                DType::I32 => {
                    let v = val_f32 as i32;
                    let coords = unravel_index(i, &out.shape);
                    let elem_offset = out.calculate_element_offset(&coords);
                    unsafe {
                        let slice = out.storage.as_mut_slice::<i32>(elem_offset, 1);
                        slice[0] = v;
                    }
                }
                DType::Bool => {
                    let v = val_f32 != 0.0;
                    let coords = unravel_index(i, &out.shape);
                    let elem_offset = out.calculate_element_offset(&coords);
                    unsafe {
                        let slice = out.storage.as_mut_slice::<bool>(elem_offset, 1);
                        slice[0] = v;
                    }
                }
                DType::F16 => {
                    let v = half::f16::from_f32(val_f32);
                    let coords = unravel_index(i, &out.shape);
                    let elem_offset = out.calculate_element_offset(&coords);
                    unsafe {
                        let slice = out.storage.as_mut_slice::<half::f16>(elem_offset, 1);
                        slice[0] = v;
                    }
                }
                DType::BF16 => {
                    let v = half::bf16::from_f32(val_f32);
                    let coords = unravel_index(i, &out.shape);
                    let elem_offset = out.calculate_element_offset(&coords);
                    unsafe {
                        let slice = out.storage.as_mut_slice::<half::bf16>(elem_offset, 1);
                        slice[0] = v;
                    }
                }
                DType::U8 => {
                    let v = val_f32 as u8;
                    let coords = unravel_index(i, &out.shape);
                    let elem_offset = out.calculate_element_offset(&coords);
                    unsafe {
                        let slice = out.storage.as_mut_slice::<u8>(elem_offset, 1);
                        slice[0] = v;
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn to_device(&self, target: Device) -> Result<Self> {
        if self.device == target {
            return Ok(self.clone());
        }
        let data = self.to_vec();
        Tensor::from_vec_device(data, self.shape.clone(), target)
    }

    pub fn clone_data(&self) -> Self {
        let vec = self.to_vec();
        Tensor::from_vec_device(vec, self.shape.clone(), self.device.clone()).unwrap()
    }

    // -----------------------------------------------------------------------
    // Backward Compatibility Bridge for Legacy Autograd
    // -----------------------------------------------------------------------

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, TensorData> {
        let legacy = self.legacy_storage.as_ref().expect("Legacy storage not initialized");
        legacy.read().expect("Poisoned lock")
    }

    pub fn data_mut(&self) -> std::sync::RwLockWriteGuard<'_, TensorData> {
        let legacy = self.legacy_storage.as_ref().expect("Legacy storage not initialized");
        legacy.write().expect("Poisoned lock")
    }

    pub fn zero_grad(&self) {
        if let Some(m) = &self.autograd_meta {
            m.lock().unwrap().grad = None;
        }
    }

    pub fn accumulate_grad(&self, delta: &Tensor) {
        self.add_storage_inplace(delta);
    }

    pub fn add_storage_inplace(&self, delta: &Tensor) {
        let n = self.numel().min(delta.numel());
        let delta_vec = delta.to_vec();
        for i in 0..n {
            let cur = self.get(i);
            self.set(i, cur + delta_vec[i]);
        }
    }

    pub fn grad_vec(&self) -> Option<Vec<f32>> {
        if let Some(g) = self.grad() {
            return Some(g.to_vec());
        }
        self.legacy_storage.as_ref().map(|g| g.read().unwrap().to_vec())
    }

    pub fn shape_str(&self) -> String {
        format!(
            "[{}]",
            self.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    // -----------------------------------------------------------------------
    // Milestone 1 Differentiable Ops on Tensor
    // -----------------------------------------------------------------------

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::binary_op(crate::dispatch::OpId::Add, self, other)?;
        if crate::autograd::is_grad_enabled() && (self.requires_grad || other.requires_grad) {
            let mut edges = Vec::new();
            let mut needs = [false, false];
            if let Some(e) = self.gradient_edge() { edges.push(e); needs[0] = true; }
            if let Some(e) = other.gradient_edge() { edges.push(e); needs[1] = true; }
            let node = Arc::new(crate::autograd::ops::AddBackward::new(self.shape().clone(), other.shape().clone(), needs, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn add_(&mut self, other: &Tensor) -> Result<()> {
        crate::dispatch::cpu::elementwise::add_(self, other)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::binary_op(crate::dispatch::OpId::Sub, self, other)?;
        if crate::autograd::is_grad_enabled() && (self.requires_grad || other.requires_grad) {
            let mut edges = Vec::new();
            let mut needs = [false, false];
            if let Some(e) = self.gradient_edge() { edges.push(e); needs[0] = true; }
            if let Some(e) = other.gradient_edge() { edges.push(e); needs[1] = true; }
            let node = Arc::new(crate::autograd::ops::SubBackward::new(self.shape().clone(), other.shape().clone(), needs, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn sub_(&mut self, other: &Tensor) -> Result<()> {
        crate::dispatch::cpu::elementwise::sub_(self, other)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::binary_op(crate::dispatch::OpId::Mul, self, other)?;
        if crate::autograd::is_grad_enabled() && (self.requires_grad || other.requires_grad) {
            let mut edges = Vec::new();
            let mut needs = [false, false];
            if let Some(e) = self.gradient_edge() { edges.push(e); needs[0] = true; }
            if let Some(e) = other.gradient_edge() { edges.push(e); needs[1] = true; }
            let node = Arc::new(crate::autograd::ops::MulBackward::new(self, other, needs, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn mul_(&mut self, other: &Tensor) -> Result<()> {
        crate::dispatch::cpu::elementwise::mul_(self, other)
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::binary_op(crate::dispatch::OpId::Div, self, other)?;
        if crate::autograd::is_grad_enabled() && (self.requires_grad || other.requires_grad) {
            let mut edges = Vec::new();
            let mut needs = [false, false];
            if let Some(e) = self.gradient_edge() { edges.push(e); needs[0] = true; }
            if let Some(e) = other.gradient_edge() { edges.push(e); needs[1] = true; }
            let node = Arc::new(crate::autograd::ops::DivBackward::new(self, other, needs, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn div_(&mut self, other: &Tensor) -> Result<()> {
        crate::dispatch::cpu::elementwise::div_(self, other)
    }

    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        let s = Tensor::from_vec_device(vec![scalar], vec![], self.device())?;
        self.div(&s)
    }

    pub fn neg(&self) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::unary_op(crate::dispatch::OpId::Neg, self)?;
        if crate::autograd::is_grad_enabled() && self.requires_grad {
            let mut edges = Vec::new();
            if let Some(e) = self.gradient_edge() { edges.push(e); }
            let node = Arc::new(crate::autograd::ops::NegBackward::new(edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn relu(&self) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::unary_op(crate::dispatch::OpId::ReLU, self)?;
        if crate::autograd::is_grad_enabled() && self.requires_grad {
            let mut edges = Vec::new();
            if let Some(e) = self.gradient_edge() { edges.push(e); }
            let node = Arc::new(crate::autograd::ops::ReLUBackward::new(self, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::matmul(self, other)?;
        if crate::autograd::is_grad_enabled() && (self.requires_grad || other.requires_grad) {
            let mut edges = Vec::new();
            let mut needs = [false, false];
            if let Some(e) = self.gradient_edge() { edges.push(e); needs[0] = true; }
            if let Some(e) = other.gradient_edge() { edges.push(e); needs[1] = true; }
            let node = Arc::new(crate::autograd::ops::MatmulBackward::new(self, other, needs, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn sum(&self, dims: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::reduction(crate::dispatch::OpId::Sum, self, dims, keepdim)?;
        if crate::autograd::is_grad_enabled() && self.requires_grad {
            let mut edges = Vec::new();
            if let Some(e) = self.gradient_edge() { edges.push(e); }
            let node = Arc::new(crate::autograd::ops::SumBackward::new(self.shape().clone(), dims.map(|d| d.to_vec()), keepdim, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn mean(&self, dims: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::reduction(crate::dispatch::OpId::Mean, self, dims, keepdim)?;
        if crate::autograd::is_grad_enabled() && self.requires_grad {
            let mut edges = Vec::new();
            if let Some(e) = self.gradient_edge() { edges.push(e); }
            let count = (self.numel() / out.numel().max(1)) as f32;
            let node = Arc::new(crate::autograd::ops::MeanBackward::new(self.shape().clone(), dims.map(|d| d.to_vec()), keepdim, count, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn log_softmax(&self, dim: isize) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::log_softmax(self, dim)?;
        if crate::autograd::is_grad_enabled() && self.requires_grad {
            let mut edges = Vec::new();
            if let Some(e) = self.gradient_edge() { edges.push(e); }
            let node = Arc::new(crate::autograd::ops::LogSoftmaxBackward::new(&out, dim, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }

    pub fn softmax(&self, dim: isize) -> Result<Tensor> {
        crate::dispatch::cpu::softmax::softmax(self, dim)
    }

    pub fn cross_entropy(&self, target: &Tensor) -> Result<Tensor> {
        let mut out = crate::dispatch::Dispatcher::cross_entropy(self, target)?;
        if crate::autograd::is_grad_enabled() && self.requires_grad {
            let mut edges = Vec::new();
            if let Some(e) = self.gradient_edge() { edges.push(e); }
            let node = Arc::new(crate::autograd::ops::CrossEntropyBackward::new(self, target, edges));
            out.set_grad_fn(node, 0);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Operator Overloads
// ---------------------------------------------------------------------------

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> {
        self.add(rhs)
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor> {
        self.sub(rhs)
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Result<Tensor> {
        self.mul(rhs)
    }
}

impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Result<Tensor>;
    fn div(self, rhs: &Tensor) -> Result<Tensor> {
        self.div(rhs)
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> {
        self.neg()
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_vec();
        let n = data.len();
        let preview_n = n.min(8);
        let preview: Vec<String> = data[..preview_n]
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect();
        write!(
            f,
            "Tensor(shape={}, strides={:?}, offset={}, dtype={}, device={}, data=[{}{}])",
            self.shape_str(),
            self.strides,
            self.storage_offset,
            self.dtype,
            self.device,
            preview.join(", "),
            if n > 8 { ", ..." } else { "" }
        )
    }
}

// ---------------------------------------------------------------------------
// Unit Tests for Strided Tensor & Views
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_strides_and_contiguity() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.strides, vec![3, 1]);
        assert_eq!(t.storage_offset, 0);
        assert!(t.is_contiguous());

        // Transpose creates non-contiguous view
        let tr = t.transpose(0, 1).unwrap();
        assert_eq!(tr.shape, vec![3, 2]);
        assert_eq!(tr.strides, vec![1, 3]);
        assert_eq!(tr.storage_offset, 0);
        assert!(!tr.is_contiguous());

        // Values match transposed layout
        assert_eq!(tr.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Contiguous copy restores contiguity
        let contig = tr.contiguous().unwrap();
        assert!(contig.is_contiguous());
        assert_eq!(contig.strides, vec![2, 1]);
        assert_eq!(contig.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_tensor_zero_copy_view_aliasing() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let tr = t.transpose(0, 1).unwrap();

        // Mutating base modifies view
        t.set(0, 99.0);
        assert_eq!(tr.get(0), 99.0);
        assert_eq!(t.version(), 1);
        assert_eq!(tr.version(), 1);
    }

    #[test]
    fn test_tensor_slice_and_expand() {
        let t = Tensor::from_vec((0..12).map(|x| x as f32).collect(), vec![3, 4]).unwrap();
        // slice rows 1..3
        let sl = t.slice(0, 1, 3, 1).unwrap();
        assert_eq!(sl.shape, vec![2, 4]);
        assert_eq!(sl.storage_offset, 4);
        assert_eq!(sl.to_vec(), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

        // expand dim 0
        let row = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
        let exp = row.expand(&[4, 3]).unwrap();
        assert_eq!(exp.shape, vec![4, 3]);
        assert_eq!(exp.strides, vec![0, 1]);
        assert_eq!(
            exp.to_vec(),
            vec![
                10.0, 20.0, 30.0,
                10.0, 20.0, 30.0,
                10.0, 20.0, 30.0,
                10.0, 20.0, 30.0
            ]
        );
    }

    #[test]
    fn test_version_counter_in_place_safety() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let saved_v = t.version();
        assert!(t.check_version(saved_v).is_ok());

        t.add_inplace_(5.0).unwrap();
        assert_eq!(t.version(), saved_v + 1);
        assert!(t.check_version(saved_v).is_err());
    }
}
