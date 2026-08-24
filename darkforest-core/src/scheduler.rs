//! Learning rate schedulers.

pub trait LRScheduler {
    fn step(&mut self) -> f32;
    fn get_lr(&self) -> f32;
}

pub struct StepLR {
    pub initial_lr: f32,
    pub current_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
    pub last_epoch: usize,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLR {
            initial_lr,
            current_lr: initial_lr,
            step_size,
            gamma,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self) -> f32 {
        self.last_epoch += 1;
        if self.last_epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

pub struct CosineAnnealingLR {
    pub initial_lr: f32,
    pub eta_min: f32,
    pub t_max: usize,
    pub current_lr: f32,
    pub last_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize, eta_min: f32) -> Self {
        CosineAnnealingLR {
            initial_lr,
            eta_min,
            t_max,
            current_lr: initial_lr,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self) -> f32 {
        self.last_epoch += 1;
        let epoch = self.last_epoch.min(self.t_max) as f32;
        let t_max = self.t_max as f32;
        let cos_val = (std::f32::consts::PI * epoch / t_max).cos();
        self.current_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + cos_val);
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

pub struct ExponentialLR {
    pub initial_lr: f32,
    pub gamma: f32,
    pub current_lr: f32,
    pub last_epoch: usize,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        ExponentialLR {
            initial_lr,
            gamma,
            current_lr: initial_lr,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self) -> f32 {
        self.last_epoch += 1;
        self.current_lr *= self.gamma;
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}
