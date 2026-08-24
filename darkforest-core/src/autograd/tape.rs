//! Thread-local grad-enabled flag.  Mirrors PyTorch's `torch.no_grad()`.

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = Cell::new(true);
}

/// Returns whether autograd is currently enabled on this thread.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Guard that disables gradient tracking for the duration of a closure.
/// Restores previous state on drop.
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    fn new() -> Self {
        let prev = is_grad_enabled();
        GRAD_ENABLED.with(|g| g.set(false));
        NoGradGuard { prev }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.prev));
    }
}

/// Run `f` with gradient tracking disabled.
/// Usage: `no_grad(|| { ... })`
pub fn no_grad<F: FnOnce() -> R, R>(f: F) -> R {
    let _guard = NoGradGuard::new();
    f()
}
