"""Dark Forest Python package.

This wrapper exposes the compiled Rust extension with a stable import surface
while keeping the core runtime under the Rust crate.
"""

import os
from importlib import import_module

cuda_roots = [
    os.environ.get("CUDA_HOME"),
    os.environ.get("CUDA_PATH"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
]

for cuda_root in cuda_roots:
    if not cuda_root:
        continue
    for dll_dir in (
        os.path.join(cuda_root, "bin"),
        os.path.join(cuda_root, "bin", "x64"),
        os.path.join(cuda_root, "lib"),
        os.path.join(cuda_root, "lib", "x64"),
    ):
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except OSError:
                pass
            if dll_dir not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

_darkforest_py = import_module("darkforest._darkforest_py")

version = _darkforest_py.version
product_name = _darkforest_py.product_name
status = _darkforest_py.status
smoke_test = _darkforest_py.smoke_test
benchmark_matrix_step = _darkforest_py.benchmark_matrix_step

__version__ = version()
__product_name__ = product_name()
__status__ = status()

__all__ = [
    "version",
    "product_name",
    "status",
    "smoke_test",
    "benchmark_matrix_step",
    "__version__",
    "__product_name__",
    "__status__",
]
