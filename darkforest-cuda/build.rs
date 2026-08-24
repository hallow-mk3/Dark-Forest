//! Build script for darkforest-cuda.
//!
//! Compiles CUDA kernels with nvcc targeting sm_120 (Blackwell, RTX 5070).
//! Set CUDA_PATH or ensure Visual Studio tools are installed when nvcc is not on PATH.

use std::env;
use std::path::{Path, PathBuf};

fn find_cuda_root() -> PathBuf {
    if let Ok(cuda_root) = env::var("CUDA_PATH") {
        return PathBuf::from(cuda_root);
    }
    if let Ok(cuda_root) = env::var("CUDA_HOME") {
        return PathBuf::from(cuda_root);
    }

    let candidates = [
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"),
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"),
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"),
    ];

    candidates.into_iter().find(|p| p.join("bin").join("nvcc.exe").exists()).unwrap_or_else(|| {
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3")
    })
}

fn find_cl_from_vs() -> Option<PathBuf> {
    let roots = [
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio"),
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    ];

    for root in roots {
        if !root.exists() {
            continue;
        }

        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    }
                }
            }

            let candidate = dir.join("VC").join("Tools").join("MSVC");
            if !candidate.exists() {
                continue;
            }

            if let Ok(entries) = std::fs::read_dir(&candidate) {
                for entry in entries.flatten() {
                    let versioned = entry.path();
                    let cl = versioned
                        .join("bin")
                        .join("Hostx64")
                        .join("x64")
                        .join("cl.exe");
                    if cl.exists() {
                        return Some(cl);
                    }
                    let cl_alt = versioned
                        .join("bin")
                        .join("Hostx86")
                        .join("x64")
                        .join("cl.exe");
                    if cl_alt.exists() {
                        return Some(cl_alt);
                    }
                }
            }
        }
    }

    None
}

fn find_cl() -> Option<PathBuf> {
    if let Ok(output) = std::process::Command::new("where.exe").arg("cl.exe").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let path = line.trim();
                if !path.is_empty() && Path::new(path).exists() {
                    return Some(PathBuf::from(path));
                }
            }
        }
    }

    find_cl_from_vs()
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(darkforest_cuda_kernels)");
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=build.rs");

    let cuda_path = find_cuda_root();
    let nvcc = cuda_path.join("bin").join("nvcc.exe");

    if !nvcc.exists() {
        println!("cargo:warning=nvcc not found at {nvcc:?}. Kernel compilation skipped.");
        println!(
            "cargo:warning=Install CUDA Toolkit 13.3 and set CUDA_PATH to enable GPU support."
        );
        return;
    }

    let cl = find_cl();
    if cl.is_none() {
        println!("cargo:warning=MSVC compiler (cl.exe) was not found in the PATH or standard Visual Studio installation directories. CUDA kernel compilation skipped.");
        println!("cargo:warning=Activate the Visual Studio C++ build environment or install CUDA build tools before enabling GPU support.");
        return;
    }

    println!("cargo:rustc-cfg=darkforest_cuda_kernels");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = PathBuf::from("kernels");
    let cl_path = cl.expect("cl.exe path missing after validation");
    let cl_dir = cl_path.parent().unwrap();
    let cuda_bin = cuda_path.join("bin");
    let mut path_entries = vec![cl_dir.to_string_lossy().to_string(), cuda_bin.to_string_lossy().to_string()];
    if let Some(existing) = env::var_os("PATH") {
        path_entries.push(existing.to_string_lossy().to_string());
    }
    let path_value = path_entries.join(";");

    let kernels = [
        "elementwise.cu",
        "softmax.cu",
        "layernorm.cu",
        "matmul.cu",
        "attention_fused.cu",
        "bf16_cast.cu",
    ];

    for kernel in &kernels {
        let src = kernel_dir.join(kernel);
        if !src.exists() {
            println!("cargo:warning=Kernel {src:?} not found, skipping.");
            continue;
        }

        let obj_name = kernel.replace(".cu", ".o");
        let obj = out_dir.join(&obj_name);

        let mut command = std::process::Command::new(&nvcc);
        command
            .env("PATH", &path_value)
            .env("CUDA_PATH", &cuda_path)
            .env("CUDA_HOME", &cuda_path)
            .args([
                "-arch=sm_120",
                "--compiler-bindir",
                cl_dir.to_str().unwrap(),
                "-O3",
                "-std=c++17",
                "-c",
                src.to_str().unwrap(),
                "-o",
                obj.to_str().unwrap(),
            ]);

        let status = command.status().expect("Failed to run nvcc");
        if !status.success() {
            panic!("nvcc compilation failed for {kernel}");
        }
    }

    let lib_name = "darkforest_kernels";
    let mut command = std::process::Command::new(&nvcc);
    command
        .env("PATH", &path_value)
        .env("CUDA_PATH", &cuda_path)
        .env("CUDA_HOME", &cuda_path)
        .args([
            "--lib",
            "-o",
            out_dir.join(format!("lib{lib_name}.a")).to_str().unwrap(),
        ])
        .args(kernels.iter().map(|k| {
            out_dir
                .join(k.replace(".cu", ".o"))
                .to_str()
                .unwrap()
                .to_string()
        }));

    let status = command.status().expect("Failed to link CUDA library");
    if !status.success() {
        panic!("Failed to create CUDA static library");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");
    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
}
