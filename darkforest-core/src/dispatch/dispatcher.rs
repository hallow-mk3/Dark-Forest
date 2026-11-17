use anyhow::{anyhow, Result};
use crate::dispatch::cpu;
use crate::tensor::{DType, Device, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpId {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    ReLU,
    Matmul,
    Sum,
    Mean,
    LogSoftmax,
    Softmax,
    CrossEntropy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Cuda,
}

impl From<&Device> for DeviceType {
    fn from(device: &Device) -> Self {
        match device {
            Device::Cpu => DeviceType::Cpu,
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => DeviceType::Cuda,
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => DeviceType::Cpu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DispatchKey {
    pub op: OpId,
    pub device: DeviceType,
    pub dtype: DType,
}

pub struct Dispatcher;

impl Dispatcher {
    pub fn binary_op(op: OpId, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let key = DispatchKey {
            op,
            device: DeviceType::from(&a.device()),
            dtype: a.dtype(),
        };

        match (key.device, key.dtype) {
            (DeviceType::Cpu, DType::F32) => match op {
                OpId::Add => cpu::elementwise::add(a, b),
                OpId::Sub => cpu::elementwise::sub(a, b),
                OpId::Mul => cpu::elementwise::mul(a, b),
                OpId::Div => cpu::elementwise::div(a, b),
                _ => Err(anyhow!("Unsupported binary op {:?}", op)),
            },
            (DeviceType::Cuda, _) => {
                Err(anyhow!("CUDA kernels will be dispatched in Step 5"))
            }
            _ => Err(anyhow!(
                "No registered kernel for key ({:?}, {:?}, {:?})",
                key.op,
                key.device,
                key.dtype
            )),
        }
    }

    pub fn unary_op(op: OpId, a: &Tensor) -> Result<Tensor> {
        let key = DispatchKey {
            op,
            device: DeviceType::from(&a.device()),
            dtype: a.dtype(),
        };

        match (key.device, key.dtype) {
            (DeviceType::Cpu, DType::F32) => match op {
                OpId::Neg => cpu::elementwise::neg(a),
                OpId::ReLU => cpu::elementwise::relu(a),
                _ => Err(anyhow!("Unsupported unary op {:?}", op)),
            },
            (DeviceType::Cuda, _) => {
                Err(anyhow!("CUDA kernels will be dispatched in Step 5"))
            }
            _ => Err(anyhow!(
                "No registered kernel for key ({:?}, {:?}, {:?})",
                key.op,
                key.device,
                key.dtype
            )),
        }
    }

    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let key = DispatchKey {
            op: OpId::Matmul,
            device: DeviceType::from(&a.device()),
            dtype: a.dtype(),
        };

        match (key.device, key.dtype) {
            (DeviceType::Cpu, DType::F32) => cpu::matmul::matmul(a, b),
            (DeviceType::Cuda, _) => {
                Err(anyhow!("CUDA kernels will be dispatched in Step 5"))
            }
            _ => Err(anyhow!(
                "No registered matmul kernel for key ({:?}, {:?})",
                key.device,
                key.dtype
            )),
        }
    }

    pub fn reduction(op: OpId, a: &Tensor, dims: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
        let key = DispatchKey {
            op,
            device: DeviceType::from(&a.device()),
            dtype: a.dtype(),
        };

        match (key.device, key.dtype) {
            (DeviceType::Cpu, DType::F32) => match op {
                OpId::Sum => cpu::reduction::sum(a, dims, keepdim),
                OpId::Mean => cpu::reduction::mean(a, dims, keepdim),
                _ => Err(anyhow!("Unsupported reduction op {:?}", op)),
            },
            (DeviceType::Cuda, _) => {
                Err(anyhow!("CUDA kernels will be dispatched in Step 5"))
            }
            _ => Err(anyhow!(
                "No registered reduction kernel for key ({:?}, {:?})",
                key.device,
                key.dtype
            )),
        }
    }

    pub fn log_softmax(a: &Tensor, dim: isize) -> Result<Tensor> {
        let key = DispatchKey {
            op: OpId::LogSoftmax,
            device: DeviceType::from(&a.device()),
            dtype: a.dtype(),
        };

        match (key.device, key.dtype) {
            (DeviceType::Cpu, DType::F32) => cpu::softmax::log_softmax(a, dim),
            _ => Err(anyhow!("Unsupported log_softmax key {:?}", key)),
        }
    }

    pub fn cross_entropy(logits: &Tensor, target: &Tensor) -> Result<Tensor> {
        let key = DispatchKey {
            op: OpId::CrossEntropy,
            device: DeviceType::from(&logits.device()),
            dtype: logits.dtype(),
        };

        match (key.device, key.dtype) {
            (DeviceType::Cpu, DType::F32) => cpu::loss::cross_entropy(logits, target),
            _ => Err(anyhow!("Unsupported cross_entropy key {:?}", key)),
        }
    }
}
