pub mod cpu;
pub mod dispatcher;
pub mod iterator;

pub use dispatcher::{DispatchKey, Dispatcher, OpId};
pub use iterator::{broadcast_shapes, effective_strides, TensorIterator};
