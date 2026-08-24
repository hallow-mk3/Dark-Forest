//! Tensor: the core data structure for Dark Forest.
//!
//! Tensors own their storage via Arc<RwLock<Vec<f32>>> so they can be
//! shared safely across the autograd graph without copying.  The dtype
//! field tracks the *logical* dtype; actual compute is always f32 on CPU
//! (f16/bf16 live in Phase 2 CUDA kernels).

use anyhow::{anyhow, Result};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// DType
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
        }
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
// Shape helper
// ---------------------------------------------------------------------------
pub type Shape = Vec<usize>;

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Compute default contiguous strides from shape (row-major).
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// TensorData  (interior mutable storage, heap-allocated)
// ---------------------------------------------------------------------------
#[derive(Debug)]
pub enum Storage {
    Cpu(Vec<f32>),
    #[cfg(feature = "cuda")]
    Cuda(darkforest_cuda::DeviceTensor),
}

#[derive(Debug)]
pub struct TensorData {
    pub storage: Storage,
}

impl TensorData {
    pub fn zeros(shape: &[usize], device: &Device) -> Self {
        match device {
            Device::Cpu => TensorData {
                storage: Storage::Cpu(vec![0.0f32; numel(shape)]),
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => TensorData {
                storage: Storage::Cuda(
                    darkforest_cuda::DeviceTensor::zeros(shape.to_vec())
                        .expect("CUDA zeros failed"),
                ),
            },
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => panic!("CUDA support is not enabled"),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize], device: &Device) -> Self {
        match device {
            Device::Cpu => TensorData {
                storage: Storage::Cpu(data),
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => TensorData {
                storage: Storage::Cuda(
                    darkforest_cuda::DeviceTensor::from_host(&data, shape.to_vec())
                        .expect("CUDA upload failed"),
                ),
            },
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => panic!("CUDA support is not enabled"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_device_tensor(device_tensor: darkforest_cuda::DeviceTensor) -> Self {
        TensorData {
            storage: Storage::Cuda(device_tensor),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        match &self.storage {
            Storage::Cpu(v) => v.clone(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(d) => d.download().expect("CUDA download failed"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_tensor(&self) -> &darkforest_cuda::DeviceTensor {
        match &self.storage {
            Storage::Cuda(d) => d,
            Storage::Cpu(_) => panic!("Expected CUDA storage on TensorData"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_tensor_mut(&mut self) -> &mut darkforest_cuda::DeviceTensor {
        match &mut self.storage {
            Storage::Cuda(d) => d,
            Storage::Cpu(_) => panic!("Expected CUDA storage on TensorData"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
/// The fundamental value type in Dark Forest.
///
/// Clone is cheap — it's Arc-sharing, not a deep copy.
/// Use `.clone_data()` for an actual deep copy of the values.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Shape,
    pub strides: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
    /// Interior-mutable storage shared with the autograd graph.
    pub storage: Arc<RwLock<TensorData>>,
    /// Accumulated gradient (None until backward sets it).
    pub grad: Option<Arc<RwLock<TensorData>>>,
    /// Whether this tensor participates in autograd.
    pub requires_grad: bool,
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
        let strides = contiguous_strides(&shape);
        Tensor {
            storage: Arc::new(RwLock::new(TensorData::zeros(&shape, &device))),
            grad: None,
            requires_grad: false,
            dtype: DType::F32,
            device,
            shape,
            strides,
        }
    }

    pub fn ones(shape: impl Into<Shape>) -> Self {
        Self::ones_device(shape, Device::Cpu)
    }

    pub fn ones_device(shape: impl Into<Shape>, device: Device) -> Self {
        let shape = shape.into();
        #[cfg(feature = "cuda")]
        if let Device::Cuda(_) = device {
            let strides = contiguous_strides(&shape);
            return Tensor {
                storage: Arc::new(RwLock::new(TensorData {
                    storage: Storage::Cuda(
                        darkforest_cuda::DeviceTensor::ones(shape.clone()).expect("CUDA ones failed"),
                    ),
                })),
                grad: None,
                requires_grad: false,
                dtype: DType::F32,
                device,
                shape,
                strides,
            };
        }
        let n = numel(&shape);
        let strides = contiguous_strides(&shape);
        Tensor {
            storage: Arc::new(RwLock::new(TensorData::from_vec(
                vec![1.0f32; n],
                &shape,
                &device,
            ))),
            grad: None,
            requires_grad: false,
            dtype: DType::F32,
            device,
            shape,
            strides,
        }
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
        Ok(Tensor {
            storage: Arc::new(RwLock::new(TensorData::from_vec(data, &shape, &device))),
            grad: None,
            requires_grad: false,
            dtype: DType::F32,
            device,
            shape,
            strides,
        })
    }

    pub fn scalar(val: f32) -> Self {
        Tensor::from_vec(vec![val], vec![1]).unwrap()
    }

    /// Kaiming uniform init (good default for weights in linear layers).
    pub fn randn(shape: impl Into<Shape>, std: f32) -> Self {
        let shape = shape.into();
        let n = numel(&shape);
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0f64, std as f64).unwrap();
        let data: Vec<f32> = (0..n).map(|_| rng.sample(dist) as f32).collect();
        Tensor::from_vec(data, shape).unwrap()
    }

    /// Uniform random in [lo, hi).
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
        self.requires_grad = true;
        self.grad = Some(Arc::new(RwLock::new(TensorData::zeros(
            &self.shape,
            &self.device,
        ))));
        self
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, TensorData> {
        self.storage.read().expect("Tensor storage poisoned")
    }

    pub fn data_mut(&self) -> std::sync::RwLockWriteGuard<'_, TensorData> {
        self.storage.write().expect("Tensor storage poisoned")
    }

    /// Read the i-th element (flat index).
    pub fn get(&self, i: usize) -> f32 {
        self.to_vec()[i]
    }

    /// Write the i-th element (flat index).
    pub fn set(&self, i: usize, val: f32) {
        let mut guard = self.data_mut();
        match &mut guard.storage {
            Storage::Cpu(v) => v[i] = val,
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => {
                drop(guard);
                let mut vec = self.to_vec();
                vec[i] = val;
                self.data_mut().storage = Storage::Cuda(
                    darkforest_cuda::DeviceTensor::from_host(&vec, self.shape.clone()).unwrap(),
                );
            }
        }
    }

    /// Return a Vec<f32> copy of the tensor's data.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data().to_vec()
    }

    pub fn to_device(&self, target: Device) -> Result<Self> {
        if self.device == target {
            return Ok(self.clone());
        }
        let data = self.to_vec();
        Tensor::from_vec_device(data, self.shape.clone(), target)
    }

    #[cfg(feature = "cuda")]
    pub fn from_cuda(device_tensor: darkforest_cuda::DeviceTensor) -> Self {
        let shape = device_tensor.shape.clone();
        let strides = contiguous_strides(&shape);
        Tensor {
            storage: Arc::new(RwLock::new(TensorData::from_device_tensor(device_tensor))),
            grad: None,
            requires_grad: false,
            dtype: DType::F32,
            device: Device::Cuda(0),
            shape,
            strides,
        }
    }

    /// Deep-copy of tensor values (not Arc-sharing).
    pub fn clone_data(&self) -> Self {
        let guard = self.data();
        let new_storage = match &guard.storage {
            Storage::Cpu(v) => Storage::Cpu(v.clone()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(d) => Storage::Cuda(d.clone_tensor().expect("DeviceTensor clone failed")),
        };
        Tensor {
            storage: Arc::new(RwLock::new(TensorData {
                storage: new_storage,
            })),
            grad: None,
            requires_grad: false,
            dtype: self.dtype,
            device: self.device.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Zero out grad accumulator.
    pub fn zero_grad(&self) {
        if let Some(ref g) = self.grad {
            let mut guard = g.write().unwrap();
            match &mut guard.storage {
                Storage::Cpu(v) => v.fill(0.0f32),
                #[cfg(feature = "cuda")]
                Storage::Cuda(d) => d.zero_().expect("CUDA zero_grad failed"),
            }
        }
    }

    /// Accumulate `delta` into the gradient buffer.
    /// DEPRECATED: do not use this when self is a grad-accumulation Tensor (ValueInner.grad).
    pub fn accumulate_grad(&self, delta: &Tensor) {
        if let Some(ref g) = self.grad {
            let mut guard = g.write().unwrap();
            match (&mut guard.storage, &delta.data().storage) {
                (Storage::Cpu(acc), Storage::Cpu(d)) => {
                    for (a, &val) in acc.iter_mut().zip(d.iter()) {
                        *a += val;
                    }
                }
                #[cfg(feature = "cuda")]
                (Storage::Cuda(acc), Storage::Cuda(d)) => {
                    acc.add_inplace(d).expect("CUDA accumulate_grad failed");
                }
                #[cfg(feature = "cuda")]
                (Storage::Cuda(acc), Storage::Cpu(d)) => {
                    let dev_d =
                        darkforest_cuda::DeviceTensor::from_host(d, delta.shape.clone()).unwrap();
                    acc.add_inplace(&dev_d)
                        .expect("CUDA accumulate_grad failed");
                }
                #[cfg(feature = "cuda")]
                (Storage::Cpu(acc), Storage::Cuda(d)) => {
                    let host_d = d.download().unwrap();
                    for (a, &val) in acc.iter_mut().zip(host_d.iter()) {
                        *a += val;
                    }
                }
            }
        }
    }

    /// Add `delta` directly into this tensor's storage (not its .grad field).
    /// Used by backward engine to accumulate gradients stored in ValueInner.grad.
    pub fn add_storage_inplace(&self, delta: &Tensor) {
        let delta_storage = delta.data();
        let mut self_storage = self.data_mut();
        match (&mut self_storage.storage, &delta_storage.storage) {
            (Storage::Cpu(acc), Storage::Cpu(d)) => {
                for (a, &val) in acc.iter_mut().zip(d.iter()) {
                    *a += val;
                }
            }
            #[cfg(feature = "cuda")]
            (Storage::Cuda(acc), Storage::Cuda(d)) => {
                acc.add_inplace(d).expect("CUDA add_storage_inplace failed");
            }
            #[cfg(feature = "cuda")]
            (Storage::Cuda(acc), Storage::Cpu(d)) => {
                let dev_d =
                    darkforest_cuda::DeviceTensor::from_host(d, delta.shape.clone()).unwrap();
                acc.add_inplace(&dev_d)
                    .expect("CUDA add_storage_inplace failed");
            }
            #[cfg(feature = "cuda")]
            (Storage::Cpu(acc), Storage::Cuda(d)) => {
                let host_d = d.download().unwrap();
                for (a, &val) in acc.iter_mut().zip(host_d.iter()) {
                    *a += val;
                }
            }
        }
    }

    pub fn grad_vec(&self) -> Option<Vec<f32>> {
        self.grad.as_ref().map(|g| g.read().unwrap().to_vec())
    }

    // -----------------------------------------------------------------------
    // Shape ops
    // -----------------------------------------------------------------------

    /// Reshape into a new shape (must have same numel).
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Tensor> {
        let new_shape = new_shape.into();
        if numel(&new_shape) != self.numel() {
            return Err(anyhow!(
                "reshape: numel mismatch {:?} → {:?}",
                self.shape,
                new_shape
            ));
        }
        let strides = contiguous_strides(&new_shape);
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            dtype: self.dtype,
            device: self.device.clone(),
            shape: new_shape,
            strides,
        })
    }

    /// Transpose last two dims (useful for attention).
    pub fn transpose_last_two(&self) -> Result<Tensor> {
        let ndim = self.ndim();
        if ndim < 2 {
            return Err(anyhow!("transpose_last_two requires ndim >= 2"));
        }
        let mut new_shape = self.shape.clone();
        new_shape.swap(ndim - 2, ndim - 1);
        // This does a data copy (not a view) — simple correctness-first impl.
        let src = self.to_vec();
        let mut dst = vec![0.0f32; src.len()];
        let rows = self.shape[ndim - 2];
        let cols = self.shape[ndim - 1];
        let batch: usize = self.shape[..ndim - 2].iter().product();
        for b in 0..batch {
            let off = b * rows * cols;
            for r in 0..rows {
                for c in 0..cols {
                    dst[off + c * rows + r] = src[off + r * cols + c];
                }
            }
        }
        Tensor::from_vec_device(dst, new_shape, self.device.clone())
    }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------

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
            "Tensor(shape={}, dtype={}, device={}, data=[{}{}])",
            self.shape_str(),
            self.dtype,
            self.device,
            preview.join(", "),
            if n > 8 { ", ..." } else { "" }
        )
    }
}
