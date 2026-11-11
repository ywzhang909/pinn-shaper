"""PINN 光束整形 API —— 与 ``gerchberg_saxton`` 同形的接口。

本模块把 :class:`pinn_shaper.Nearfield_PINN` / :class:`pinn_shaper.Farfield_PINN`
包装成一个与 AO-shaping 的
``ao_shaping.algorithm.signal_processing.gerchberg_saxton.gerchberg_saxton``
**同签名、同返回约定**的函数 :func:`pinn_shaping`，因此凡是用 GS 产生整形相位
的下游代码（SLM 下发、评估、报告）可以原样替换成 PINN，只需把光源/目标以
**振幅**（intensity 的平方根）给出即可。

分辨率对齐
----------
GS 用 ``source_amplitude`` 的像素网格 + ``cell_spacing`` 定义物理/角域。本模块
沿用完全相同的约定，并据此把像素数组映射到 PINN 的物理坐标：

* 源面（与 GS 一致）::

      extent_source = N * cell_spacing            # N = source_amplitude.shape[0]
      M             = extent_source / 2
      x_i = (i - (N - 1) / 2) * cell_spacing      # 对称、像素中心，间距恰为 dx

* 近场 (``propagation="asm"``，有限距离)：角谱传播不改变网格，目标平面与源面
  **同一分辨率**，故 ``extent_target = extent_source``；方向余弦域不限。

* 远场 (``propagation="fft"``，单次 FFT)：目标平面是角域，像素网格的
  **奈奎斯特方向余弦**为::

      alpha_max = min(wavelength / (2 * cell_spacing), 1.0)

  目标振幅数组的像素 ``[-N/2, N/2]`` 线性映射到
  ``alpha ∈ [-alpha_max, alpha_max]``。这样 PINN 训练用的 ``(α, β)`` 网格与
  FFT / 衍射仿真（diffractsim ``get_farfield``）的采样域**逐点对齐**；若源网格
  比 ``λ/2`` 更细（``alpha_max`` 会超过 1），则截断到物理极限 ``sinθ ≤ 1``。

注意：PINN 的解在方形归一化域 ``[-M, M]²`` 上定义，因此要求
``source_amplitude`` 为**方形**数组（``Ny == Nx``）；GS 允许矩形，PINN 不支持。

Example
-------
>>> from pinn_shaper.shaping import pinn_shaping
>>> res = pinn_shaping(
...     source_amplitude=source_amp,      # (N, N), 振幅
...     target_amplitude=target_amp,      # (N, N), 振幅
...     iterations=2000,
...     cell_spacing=8e-6,
...     distance=0.1,
...     wavelength=1064e-9,
...     propagation="asm",
... )
>>> slm_phase = res.phase                 # (N, N) 弧度, 与 GS 的 result.phase 同义
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import torch

from .farfield_shaper import Farfield_PINN
from .nearfield_shaper import Nearfield_PINN
from .util import create_interpolator

__all__ = [
    "PINNShapingResult",
    "pinn_shaping",
    "evaluate_shaping",
    "angular_spectrum_propagate",
]


# ----------------------------------------------------------------------
# 结果容器 —— 字段名与 gerchberg_saxton.GSResult 保持一致
# ----------------------------------------------------------------------
@dataclass
class PINNShapingResult:
    """PINN 整形的结果容器（与 ``GSResult`` 同名字段）。

    Attributes:
        phase: 为 SLM 算出的相位图形（弧度；未包裹，与 GS 的 ``phase`` 同义）
        amplitude: 目标面上的最终振幅（与 ``phase`` 同一网格）
        error_history: 每次评估点的 MSE（``|amplitude|`` 与 ``target_amplitude``）
        iterations: 实际执行的优化步数
        converged: 是否在 ``error_threshold`` 下提前收敛
        mse: 最终 MSE（同 ``gerchberg_saxton`` 的定义）
        rel_rms: 最小二乘重标定后的相对 RMS（推荐用这个跨尺度比较）
        efficiency: 目标支撑域内能量 / 源面总能量（论文的 η）
        solver: 训练好的 PINN 实例（便于继续训练或取样条）
        target_amplitude: 传入的目标振幅（便于绘图对比）
    """

    phase: npt.NDArray[np.floating]
    amplitude: npt.NDArray[np.floating]
    error_history: list[float]
    iterations: int
    converged: bool
    mse: float = float("nan")
    rel_rms: float = float("nan")
    efficiency: float = float("nan")
    solver: object | None = None
    target_amplitude: npt.NDArray[np.floating] | None = field(default=None)
    loss_history: list[float] = field(default_factory=list)


# ----------------------------------------------------------------------
# 传播工具（numpy 实现，与 gerchberg_saxton 的 ASM 逐位同构）
# ----------------------------------------------------------------------
@lru_cache(maxsize=32)
def _compute_propagator(
    ny: int, nx: int, dx: float, z: float, wavelength: float
) -> npt.NDArray[np.complexfloating]:
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy)
    kx, ky = 2 * np.pi * FX, 2 * np.pi * FY
    kz_sq = k**2 - kx**2 - ky**2
    H = np.exp(1j * np.sqrt(np.maximum(kz_sq, 0)) * z)
    H[kz_sq < 0] = 0.0  # 倏逝波
    return np.fft.ifftshift(H)


def angular_spectrum_propagate(
    field: npt.NDArray, dx: float, z: float, wavelength: float
) -> npt.NDArray[np.complexfloating]:
    """角谱法传播复光场（与 GS 模块的实现一致）。"""
    if field.ndim != 2:
        raise ValueError(f"Field must be 2D array, got {field.ndim}D")
    F = np.fft.fft2(field)
    H = _compute_propagator(field.shape[0], field.shape[1], dx, z, wavelength)
    return np.fft.ifft2(F * H)


def _propagate_amplitude(
    phase: npt.NDArray[np.floating],
    source_amplitude: npt.NDArray[np.floating],
    cell_spacing: float,
    distance: float,
    wavelength: float,
    propagation: str,
) -> npt.NDArray[np.floating]:
    """给定相位，返回目标面振幅 ``|E_target|``。"""
    field = source_amplitude.astype(np.complex128) * np.exp(1j * phase)
    if propagation == "asm":
        out = angular_spectrum_propagate(field, cell_spacing, distance, wavelength)
    elif propagation == "fft":
        # 酉变换（除以 N），保持 Parseval 能量守恒，使振幅尺度与源面可比
        out = np.fft.fftshift(np.fft.fft2(field)) / np.sqrt(field.size)
    else:
        raise ValueError(f"propagation must be 'asm' or 'fft', got {propagation!r}")
    return np.abs(out)


def _shaping_metrics(
    amplitude: npt.NDArray[np.floating],
    target_amplitude: npt.NDArray[np.floating],
    source_amplitude: npt.NDArray[np.floating],
    support_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """返回 ``(mse, rel_rms, capture_efficiency)``。

    * ``mse``：与 ``gerchberg_saxton`` 同定义 ``mean((|C| - target)²)``（受
      绝对幅度尺度影响，近场/远场两种传播下的尺度约定见模块文档）。
    * ``rel_rms``：先做最小二乘全局重标定，再算相对 RMS —— **跨模式、跨
      尺度都可比**，是论文表格里用的指标。
    * ``capture_efficiency``：落进目标支撑域（``target >= thresh * max``）
      的能量占源面总能量之比，即论文的 η。
    """
    amp = np.asarray(amplitude, dtype=np.float64)
    tgt = np.asarray(target_amplitude, dtype=np.float64)
    mse = float(np.mean((amp - tgt) ** 2))

    p = amp.ravel()
    t = tgt.ravel()
    denom = float(np.sum(p * p))
    scale = float(np.sum(p * t) / denom) if denom > 0 else 0.0
    num = float(np.mean((scale * p - t) ** 2))
    den = float(np.mean(t**2))
    rel_rms = float(np.sqrt(num / den)) if den > 0 else float("inf")

    src = np.asarray(source_amplitude, dtype=np.float64)
    support = tgt >= support_threshold * (float(tgt.max()) if tgt.max() > 0 else 1.0)
    eff = float(np.sum(amp[support] ** 2) / (np.sum(src**2) + 1e-30))
    return mse, rel_rms, eff


# ----------------------------------------------------------------------
# 设备上下文
# ----------------------------------------------------------------------
class _default_device:
    """临时把 torch 默认设备切到 *device*，退出时恢复。"""

    def __init__(self, device: torch.device):
        self.device = device
        self.prev = "cpu"

    def __enter__(self):
        try:
            self.prev = str(torch.get_default_device())
        except AttributeError:  # 老版本 torch
            self.prev = "cpu"
        torch.set_default_device(self.device)
        return self

    def __exit__(self, *exc):
        torch.set_default_device(self.prev)
        return False


# ----------------------------------------------------------------------
# 主接口
# ----------------------------------------------------------------------
def pinn_shaping(
    source_amplitude: npt.NDArray[np.floating],
    target_amplitude: npt.NDArray[np.floating],
    iterations: int = 3000,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    error_threshold: float | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
    phase_callback: Callable[[int, npt.NDArray[np.floating]], None] | None = None,
    propagation: str = "asm",
    *,
    hidden_layers: int = 3,
    num_features: int = 150,
    phi_scale: float | None = None,
    symmetry: str | None = None,
    constrain: bool = False,
    constrain_fn: Callable | None = None,
    integration_points: int = 2000,
    num_domain: int = 1500,
    num_boundary: int = 100,
    residual_power: float = 1.5,
    loss_weights: tuple[float, float, float] = (100.0, 1.0, 1.0),
    lr: float = 1e-3,
    lbfgs_iterations: int = 0,
    lbfgs_lr: float = 1.0,
    noise: float = 0.001,
    eval_interval: int = 100,
    callback_interval: int = 100,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
    verbose: bool = True,
) -> PINNShapingResult:
    """用 PINN 计算使目标面产生期望强度的源面相位（GS 的 PINN 替代品）。

    参数名称与语义刻意与
    :func:`ao_shaping.algorithm.signal_processing.gerchberg_saxton.gerchberg_saxton`
    保持一致（``source_amplitude`` / ``target_amplitude`` / ``iterations`` /
    ``cell_spacing`` / ``distance`` / ``wavelength`` / ``error_threshold`` /
    ``progress_callback`` / ``phase_callback`` / ``propagation``），返回值同样是
    一个带 ``phase`` / ``amplitude`` / ``error_history`` / ``iterations`` /
    ``converged`` 的结果对象。

    Args:
        source_amplitude: 二维方形数组，SLM/源面振幅约束（强度平方根）。
        target_amplitude: 二维方形数组，目标面期望振幅。近场为目标面强度
            平方根；远场为方向余弦 ``α, β ∈ [-alpha_max, alpha_max]`` 上的
            远场振幅平方根。
        iterations: Adam 优化步数（GS 里是 GS 循环次数；PINN 需要更多步，
            默认 3000；论文日程见 README）。
        cell_spacing: 像素间距（米），决定物理/角域尺度（分辨率对齐的关键）。
        distance: 传播距离（米）。仅 ``propagation="asm"`` 使用。
        wavelength: 波长（米）。
        error_threshold: 可选收敛阈值（MSE）。
        progress_callback: ``(iter, mse)`` 回调，每 ``eval_interval`` 调用。
        phase_callback: ``(iter, phase)`` 回调，每 ``callback_interval`` 调用。
        propagation: ``"asm"``（近场/有限距离，默认）或 ``"fft"``（远场）。
        hidden_layers, num_features: FCNN 结构。
        phi_scale: 网络输出缩放；``None`` 时近场 30、远场 8（论文默认）。
        symmetry: ``None`` / ``"x"`` / ``"y"`` / ``"xy"``，几何对称缩减。
        constrain: 是否给网络加解析硬约束（远场星形常用 ``r²``）。
        constrain_fn: 硬约束函数 ``g(x)``。
        integration_points: 功率归一化采样点数。
        num_domain, num_boundary: 每次采样的内部 / 边界点数。
        residual_power: PDE 残差幂次。
        loss_weights: ``(w_int, w_bx, w_by)``。
        lr: Adam 学习率。
        lbfgs_iterations: Adam 之后的 L-BFGS 步数（0 表示不跑）。
        lbfgs_lr: L-BFGS 初始步长。
        noise: 采样抖动。
        eval_interval: 计算 MSE / 回调的间隔。
        callback_interval: ``phase_callback`` 的间隔。
        device: ``"cuda"`` / ``"cpu"`` / ``torch.device``；``None`` 自动选择。
        dtype: torch 数据类型。
        seed: 随机种子。
        verbose: 是否打印进度。

    Returns:
        :class:`PINNShapingResult`，``result.phase`` 即要下发的相位（弧度）。

    Raises:
        ValueError: 输入非方形 / 维度不对 / 物理参数非法时。
    """
    # ---------------- 输入校验 ----------------
    if source_amplitude.ndim != 2 or target_amplitude.ndim != 2:
        raise ValueError("source_amplitude and target_amplitude must be 2D arrays")
    if source_amplitude.shape != target_amplitude.shape:
        raise ValueError(
            "Source and target shapes must match: "
            f"{source_amplitude.shape} vs {target_amplitude.shape}"
        )
    ny, nx = source_amplitude.shape
    if ny != nx:
        raise ValueError(
            "PINN shaping requires a square grid (Ny == Nx); got "
            f"{ny} x {nx}. Crop/pad the source and target to a common square."
        )
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if cell_spacing <= 0 or distance <= 0 or wavelength <= 0:
        raise ValueError("cell_spacing, distance and wavelength must be positive")
    if propagation not in ("asm", "fft"):
        raise ValueError(f"propagation must be 'asm' or 'fft', got {propagation!r}")

    # ---------------- 分辨率对齐 ----------------
    N = nx
    dx = float(cell_spacing)
    extent_source = N * dx
    M = extent_source / 2.0
    # 像素中心坐标：间距恰为 dx，且关于 0 对称
    axis = (torch.arange(N, dtype=dtype) - (N - 1) / 2.0) * dx

    if propagation == "asm":
        extent_target = extent_source
        T = extent_target / 2.0
        target_region = [-T, T, -T, T]
        alpha_max = None
        if phi_scale is None:
            phi_scale = 30.0
    else:
        alpha_max = min(wavelength / (2.0 * dx), 1.0)
        target_region = [-alpha_max, alpha_max, -alpha_max, alpha_max]
        extent_target = None
        T = None
        if phi_scale is None:
            phi_scale = 8.0

    # ---------------- 设备 / 种子 ----------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if verbose:
        print(
            f"[pinn_shaping] N={N} dx={dx * 1e6:.2f}µm extent={extent_source * 1e3:.3f}mm "
            f"z={distance * 1e3:.1f}mm λ={wavelength * 1e9:.0f}nm propagation={propagation}"
            + (f" alpha_max={alpha_max:.4f}" if alpha_max is not None else "")
        )

    with _default_device(dev):
        # 插值器（振幅）建立在该设备上
        src_t = torch.as_tensor(np.asarray(source_amplitude, dtype=np.float32), device=dev)
        tgt_t = torch.as_tensor(np.asarray(target_amplitude, dtype=np.float32), device=dev)
        src_interp = create_interpolator(
            src_t, mode="bilinear", region=[-M, M, -M, M], align_corners=True
        )
        tgt_interp = create_interpolator(
            tgt_t, mode="bilinear", region=target_region, align_corners=True
        )

        # PINN 的 I_fun/E_fun/Irad_fun 是**强度**；接口收的是振幅，故平方
        if propagation == "asm":
            def I_fun(x, y):
                return src_interp(x, y) ** 2

            def E_fun(x, y):
                return tgt_interp(x, y) ** 2

            solver = Nearfield_PINN(
                I_fun, E_fun, extent_source, extent_target, distance, wavelength,
                phi_scale=phi_scale, integration_points=integration_points,
                hidden_layers=hidden_layers, num_features=num_features,
                constrain=constrain, symmetry=symmetry, constrain_fn=constrain_fn,
            )
        else:
            def I_fun(x, y):
                return src_interp(x, y) ** 2

            def Irad_fun(a, b):
                return tgt_interp(a, b) ** 2

            solver = Farfield_PINN(
                I_fun, Irad_fun, extent_source, wavelength,
                phi_scale=phi_scale, integration_points=integration_points,
                hidden_layers=hidden_layers, num_features=num_features,
                constrain=constrain, symmetry=symmetry, constrain_fn=constrain_fn,
            )

        # ---------------- 优化 ----------------
        opt = torch.optim.Adam(solver.net.parameters(), lr=lr)
        lw = loss_weights
        error_history: list[float] = []
        loss_history: list[float] = []
        steps_done = 0
        converged = False

        def _phase_now() -> npt.NDArray[np.floating]:
            X, Y = torch.meshgrid(axis.to(dev), axis.to(dev), indexing="xy")
            with torch.no_grad():
                phi = solver.evaluate_phi(X, Y).reshape(N, N)
            return phi.detach().cpu().numpy()

        def _eval_mse() -> float:
            amp = _propagate_amplitude(
                _phase_now(), np.asarray(source_amplitude, np.float64),
                dx, distance, wavelength, propagation,
            )
            mse, _, _ = _shaping_metrics(
                amp, np.asarray(target_amplitude, np.float64),
                np.asarray(source_amplitude, np.float64),
            )
            return mse

        for it in range(iterations):
            opt.zero_grad()
            xD = solver.sample_interior_points(num_domain, noise)
            xBx = solver.sample_boundary_x(num_boundary, noise)
            xBy = solver.sample_boundary_y(num_boundary, noise)

            L_int = solver.pde_residual(xD, residual_power).mean()
            L_bx = (
                solver.boundary_loss_x(xBx).pow(2).mean()
                if xBx.numel() else torch.zeros((), device=dev)
            )
            L_by = (
                solver.boundary_loss_y(xBy).pow(2).mean()
                if xBy.numel() else torch.zeros((), device=dev)
            )
            loss = lw[0] * L_int + lw[1] * L_bx + lw[2] * L_by
            loss.backward()
            opt.step()
            steps_done += 1

            if it % eval_interval == 0:
                mse = _eval_mse()
                error_history.append(mse)
                loss_history.append(float(loss.detach().item()))
                if progress_callback is not None:
                    progress_callback(it, mse)
                if verbose and it % max(eval_interval, 1) == 0:
                    print(f"[pinn_shaping] adam iter {it:5d}  mse={mse:.6e}")
                if error_threshold is not None and mse < error_threshold:
                    converged = True
                    if verbose:
                        print(f"[pinn_shaping] converged at iter {it} (mse={mse:.6e})")
                    break
                if phase_callback is not None and it % max(callback_interval, 1) == 0:
                    phase_callback(it, _phase_now())

        if lbfgs_iterations > 0:
            if verbose:
                print(f"[pinn_shaping] L-BFGS {lbfgs_iterations} steps")
            solver.run_phase(
                optimizer_type="lbfgs",
                num_domain=max(num_domain, 1500),
                num_boundary=max(num_boundary, 100),
                residual_power=residual_power,
                loss_weights=loss_weights,
                effective_noise=0.0,
                num_test_interior=1000,
                num_test_boundary=200,
                max_iterations_lbfgs=lbfgs_iterations,
                lr_lbfgs=lbfgs_lr,
            )
            steps_done += lbfgs_iterations
            mse = _eval_mse()
            error_history.append(mse)
            if progress_callback is not None:
                progress_callback(iterations, mse)

        # ---------------- 最终结果 ----------------
        phase = _phase_now()
        amplitude = _propagate_amplitude(
            phase, np.asarray(source_amplitude, np.float64),
            dx, distance, wavelength, propagation,
        )
        mse, rel_rms, eff = _shaping_metrics(
            amplitude, np.asarray(target_amplitude, np.float64),
            np.asarray(source_amplitude, np.float64),
        )
        if verbose:
            print(
                f"[pinn_shaping] done: mse={mse:.6e} rel_rms={rel_rms:.4%} "
                f"efficiency={eff:.4%}"
            )

    return PINNShapingResult(
        phase=phase,
        amplitude=amplitude,
        error_history=error_history if error_history else [mse],
        iterations=steps_done,
        converged=converged or (error_threshold is not None and mse < error_threshold),
        mse=mse,
        rel_rms=rel_rms,
        efficiency=eff,
        solver=solver,
        target_amplitude=np.asarray(target_amplitude, dtype=np.float64),
        loss_history=loss_history,
    )


def evaluate_shaping(
    computed_phase: npt.NDArray[np.floating],
    source_amplitude: npt.NDArray[np.floating],
    target_amplitude: npt.NDArray[np.floating],
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    propagation: str = "asm",
) -> dict[str, float]:
    """衡量任意相位（PINN 或 GS）的整形质量，字段与 GS 模块一致。

    Returns:
        ``{"mse", "nmse", "correlation", "efficiency"}``，与
        ``gerchberg_saxton.calculate_reconstruction_error`` 同名同义。
    """
    amp = _propagate_amplitude(
        np.asarray(computed_phase, float), np.asarray(source_amplitude, float),
        cell_spacing, distance, wavelength, propagation,
    )
    tgt = np.asarray(target_amplitude, float)
    tgt_n = tgt / (tgt.max() + 1e-12)
    amp_n = amp / (amp.max() + 1e-12)
    mse = float(np.mean((amp_n - tgt_n) ** 2))
    nmse = float(mse / (np.mean(tgt_n**2) + 1e-12))
    corr = float(np.corrcoef(amp_n.ravel(), tgt_n.ravel())[0, 1])
    eff = float(np.sum(amp**2) / (np.sum(np.asarray(source_amplitude, float) ** 2) + 1e-30))
    return {"mse": mse, "nmse": nmse, "correlation": corr, "efficiency": eff}
