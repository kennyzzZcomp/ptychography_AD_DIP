"""Experimental fixed-raster adjoint retaining PyTorch 2.5.1 CUDA sum order.

No sorting, FP64, support, subset, altered optical model or gradient detach.
Independent geometry-driven kernel; reduction order is audited against the
installed version's Indexing.cu stride-one path. Not a general index operator.
Uses PyTorch's existing Windows NVRTC DLL, not an installed compiler toolchain.
The original optics/default solvers do not import or use this module.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
from pathlib import Path
import threading
import time

import torch


_SOURCE = r'''
extern "C" __global__ void raster_ordered_adjoint(
    const float* g, float* out, int oh, int ow, int r0, int c0,
    int nr, int nc, int sr, int sc, int n) {
    const int lane = threadIdx.x & 31;
    const int pixel = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (pixel >= oh * ow) return;
    const int y = pixel / ow - r0, x = pixel % ow - c0;
    float vr = 0.0f, vi = 0.0f;
    if (y >= 0 && x >= 0 && y < n+(nr-1)*sr && x < n+(nc-1)*sc) {
        const int rl = max(0, (y-n+sr)/sr), rh = min(nr-1, y/sr);
        const int cl = max(0, (x-n+sc)/sc), ch = min(nc-1, x/sc);
        const int width = ch-cl+1, count = (rh-rl+1)*width;
        if (width > 0 && rh >= rl) {
            const int groups = count/32;
            for (int k = 0; k < groups; ++k) {
                const int ordinal = 32*k+lane;
                const int r = rl+ordinal/width, c = cl+ordinal%width;
                const int at = ((r*nc+c)*n+y-r*sr)*n+x-c*sc;
                vr += g[2*at]; vi += g[2*at+1];
            }
            // All lanes enter this same branch; no active-mask reduction.
            __syncwarp();
            for (int offset = 16; offset > 0; offset /= 2) {
                vr += __shfl_down_sync(0xffffffff, vr, offset);
                vi += __shfl_down_sync(0xffffffff, vi, offset);
            }
            if (lane == 0) {
                for (int ordinal = groups*32; ordinal < count; ++ordinal) {
                    const int r = rl+ordinal/width, c = cl+ordinal%width;
                    const int at = ((r*nc+c)*n+y-r*sr)*n+x-c*sc;
                    vr += g[2*at]; vi += g[2*at+1];
                }
            }
        }
    }
    if (lane == 0) { out[2*pixel] = 0.0f+vr; out[2*pixel+1] = 0.0f+vi; }
}
'''

_CACHE = {}
_LOCK = threading.Lock()


def validate_geometry(geometry):
    h, w = geometry.object_shape
    values = (h, w, geometry.rows, geometry.cols, geometry.row_step,
              geometry.col_step, geometry.patch_size)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Require positive integer raster geometry")
    if geometry.origin_row < 0 or geometry.origin_col < 0:
        raise ValueError("Negative raster origin")
    eh, ew = geometry.extent
    if geometry.origin_row+eh > h or geometry.origin_col+ew > w:
        raise ValueError("Raster outside object")
    if max(h*w, 2*geometry.count*geometry.patch_size**2) >= 2**31:
        raise ValueError("Native-order pilot uses checked 32-bit element offsets")


class _Kernel:
    def __init__(self, device):
        if os.name != "nt" or torch.__version__.split("+")[0] != "2.5.1" or torch.version.cuda != "12.1":
            raise RuntimeError("Native-order NVRTC pilot currently requires Windows torch2.5.1 CUDA12.1")
        started = time.perf_counter()
        library = Path(torch.__file__).resolve().parent/"lib"/"nvrtc64_120_0.dll"
        if not library.is_file():
            raise RuntimeError("PyTorch's NVRTC library unavailable; do not install or silently substitute")
        self.library_directory = os.add_dll_directory(str(library.parent))
        self.nvrtc = ct.CDLL(str(library))
        self.driver = ct.WinDLL("nvcuda.dll")
        vp, cp, ip, sp = ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_size_t)
        signatures = {
            "nvrtcVersion": [ip, ip], "nvrtcCreateProgram": [ct.POINTER(vp), cp, cp, ct.c_int, ct.POINTER(cp), ct.POINTER(cp)],
            "nvrtcCompileProgram": [vp, ct.c_int, ct.POINTER(cp)], "nvrtcGetProgramLogSize": [vp, sp],
            "nvrtcGetProgramLog": [vp, vp], "nvrtcGetPTXSize": [vp, sp], "nvrtcGetPTX": [vp, vp],
            "nvrtcDestroyProgram": [ct.POINTER(vp)],
        }
        for name, args in signatures.items():
            function = getattr(self.nvrtc, name); function.argtypes = args; function.restype = ct.c_int
        signatures = {"cuModuleLoadDataEx": [ct.POINTER(vp), vp, ct.c_uint, vp, vp],
                      "cuModuleGetFunction": [ct.POINTER(vp), vp, cp],
                      "cuLaunchKernel": [vp]+[ct.c_uint]*7+[vp, ct.POINTER(vp), vp]}
        for name, args in signatures.items():
            function = getattr(self.driver, name); function.argtypes = args; function.restype = ct.c_int
        major, minor = ct.c_int(), ct.c_int()
        self.check(self.nvrtc.nvrtcVersion(ct.byref(major), ct.byref(minor)), "NVRTC version")
        if (major.value, minor.value) != (12, 1):
            raise RuntimeError("Unexpected NVRTC version")
        self.device = torch.device(device)
        # CUDA lazy initialization/device-property queries alone do not ensure
        # a current primary context for Driver API module loading on Windows.
        context_guard = torch.empty(1, device=self.device)
        architecture = torch.cuda.get_device_capability(self.device)
        self.options = (f"--gpu-architecture=compute_{architecture[0]}{architecture[1]}", "--std=c++11", "--fmad=false")
        options = (cp*len(self.options))(*(option.encode() for option in self.options))
        program = vp()
        self.check(self.nvrtc.nvrtcCreateProgram(ct.byref(program), _SOURCE.encode(), b"raster_order.cu", 0, None, None), "create program")
        try:
            result = self.nvrtc.nvrtcCompileProgram(program, len(options), options)
            size = ct.c_size_t()
            self.check(self.nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size)), "get compile log size")
            log = ct.create_string_buffer(size.value)
            self.check(self.nvrtc.nvrtcGetProgramLog(program, log), "get compile log")
            self.compile_log = log.value.decode(errors="replace")
            if result:
                raise RuntimeError(f"NVRTC compile error {result}: {self.compile_log}")
            self.check(self.nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), "get PTX size")
            ptx = ct.create_string_buffer(size.value)
            self.check(self.nvrtc.nvrtcGetPTX(program, ptx), "get PTX")
        finally:
            self.check(self.nvrtc.nvrtcDestroyProgram(ct.byref(program)), "destroy program")
        # PyTorch has already selected/initialized this device's primary context.
        self.module, self.function = vp(), vp()
        self.check(self.driver.cuModuleLoadDataEx(ct.byref(self.module), ptx, 0, None, None), "load PTX module")
        self.check(self.driver.cuModuleGetFunction(ct.byref(self.function), self.module, b"raster_ordered_adjoint"), "get kernel")
        # Eagerly include deferred driver JIT/function loading in cold cost.
        probe = torch.zeros(1, 1, 1, dtype=torch.complex64, device=self.device)
        from .raster_optics import RasterGeometry
        self.launch(probe, RasterGeometry(0, 0, 1, 1, 1, 1, 1, (1, 1)))
        torch.cuda.synchronize(self.device)
        self.cold_initialization_seconds = time.perf_counter()-started

    @staticmethod
    def check(code, action):
        if code:
            raise RuntimeError(f"CUDA/NVRTC {action} failed with code {code}")

    def launch(self, grad, geometry):
        output = torch.empty(geometry.object_shape, dtype=grad.dtype, device=grad.device)
        values = [ct.c_void_p(grad.data_ptr()), ct.c_void_p(output.data_ptr())]+[ct.c_int(value) for value in
                  (*geometry.object_shape, geometry.origin_row, geometry.origin_col, geometry.rows,
                   geometry.cols, geometry.row_step, geometry.col_step, geometry.patch_size)]
        pointers = (ct.c_void_p*len(values))(*(ct.cast(ct.byref(value), ct.c_void_p) for value in values))
        grid = (output.numel()+7)//8
        self.check(self.driver.cuLaunchKernel(self.function, grid, 1, 1, 256, 1, 1, 0,
                                              ct.c_void_p(torch.cuda.current_stream(grad.device).cuda_stream),
                                              pointers, None), "launch raster adjoint")
        # Both tensors are allocated/used on PyTorch's current stream; no foreign
        # stream, synchronization, retained tensor pointer, or manual allocation.
        return output


def prepare_native_order(device):
    """Compile/load once; callers must report this cold cost, not hide it."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native-order pilot is CUDA-only")
    with torch.cuda.device(device):
        key = torch.cuda.current_device()
        with _LOCK:
            if key not in _CACHE:
                _CACHE[key] = _Kernel(torch.device("cuda", key))
        kernel = _CACHE[key]
    return {"cold_initialization_seconds": kernel.cold_initialization_seconds,
            "nvrtc_version": "12.1", "compile_options": list(kernel.options),
            "kernel_source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
            "deployment_scope": "Windows torch2.5.1 CUDA12.1 complex64 fixed integer raster",
            "no_runtime_index_sort": True, "no_fast_math": True}


def native_order_adjoint(gradient, geometry):
    validate_geometry(geometry)
    if gradient.device.type != "cuda" or gradient.dtype != torch.complex64 \
       or tuple(gradient.shape) != (geometry.count, geometry.patch_size, geometry.patch_size):
        raise ValueError("Expected CUDA complex64 complete raster window gradient")
    with torch.cuda.device(gradient.device):
        prepare_native_order(gradient.device)
        canonical = gradient.resolve_conj().resolve_neg().contiguous()
        return _CACHE[torch.cuda.current_device()].launch(canonical, geometry)


class _NativeOrderTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradient, geometry):
        ctx.geometry = geometry
        return native_order_adjoint(gradient, geometry)

    @staticmethod
    def backward(ctx, object_gradient):
        # Exact transpose of the linear adjoint; supports higher-order queries.
        return ctx.geometry.patches(object_gradient), None


class _NativeOrderExtract(torch.autograd.Function):
    @staticmethod
    def forward(ctx, obj, geometry):
        validate_geometry(geometry)
        if obj.device.type != "cuda" or obj.dtype != torch.complex64 or tuple(obj.shape) != geometry.object_shape:
            raise ValueError("Expected CUDA complex64 object matching fixed raster")
        ctx.geometry = geometry
        return geometry.patches(obj)

    @staticmethod
    def backward(ctx, gradient):
        return _NativeOrderTranspose.apply(gradient, ctx.geometry), None


def native_order_patches(obj, geometry):
    return _NativeOrderExtract.apply(obj, geometry)
