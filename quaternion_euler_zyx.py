#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四元数与欧拉角（Z-Y-X 内旋顺序）相互转换。

约定：
- 欧拉角 (rz, ry, rx)：先绕固定坐标系的 Z 轴转 rz，再绕**转动后**的 Y' 轴转 ry，
  再绕**转动后**的 X'' 轴转 rx（内旋 / 体轴顺序 Z-Y-X）。
  等价旋转矩阵：R = Rz(rz) @ Ry(ry) @ Rx(rx)。
- 四元数 q = (w, x, y, z)，Hamilton 约定，单位四元数；与 R 的对应关系为
  对列向量 v：v' = R @ v 与 v' = q * v * q*（纯四元数嵌入）一致。
- 角度默认弧度；若使用角度制，将 degrees=True。
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)


def _rad(deg: bool, a: float) -> float:
    return math.radians(a) if deg else a


def _deg_out(deg: bool, a: float) -> float:
    return math.degrees(a) if deg else a


def quat_normalize(q: Quat) -> Quat:
    w, x, y, z = q
    n = math.hypot(w, math.hypot(x, math.hypot(y, z)))
    if n < 1e-15:
        raise ValueError("四元数范数过小，无法归一化")
    return (w / n, x / n, y / n, z / n)


def quat_mult(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def euler_zyx_to_quaternion(
    rz: float, ry: float, rx: float, *, degrees: bool = False
) -> Quat:
    """
    内旋 Z-Y-X：R = Rz(rz) Ry(ry) Rx(rx) 对应的单位四元数 q = qz * qy * qx。
    """
    rz = _rad(degrees, rz)
    ry = _rad(degrees, ry)
    rx = _rad(degrees, rx)

    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)

    qz: Quat = (cz, 0.0, 0.0, sz)
    qy: Quat = (cy, 0.0, sy, 0.0)
    qx: Quat = (cx, sx, 0.0, 0.0)
    return quat_normalize(quat_mult(quat_mult(qz, qy), qx))


def quaternion_to_euler_zyx(
    q: Quat, *, degrees: bool = False
) -> Tuple[float, float, float]:
    """
    单位四元数 -> (rz, ry, rx)，与 euler_zyx_to_quaternion 互逆（除万向节锁附近的多值性）。
    """
    w, x, y, z = quat_normalize(q)

    # 旋转矩阵 R 元素（与 Hamilton (w,x,y,z) 标准公式一致）
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    # R = Rz*Ry*Rx  =>  r20 = -sin(ry)
    s = -r20
    s = max(-1.0, min(1.0, s))  # 数值夹紧
    ry = math.asin(s)

    if abs(math.cos(ry)) > 1e-8:
        rz = math.atan2(r10, r00)
        rx = math.atan2(r21, r22)
    else:
        # 万向节锁：ry ≈ ±pi/2，令 rx = 0，rz 由 atan2 组合得到
        rx = 0.0
        rz = math.atan2(-r01, r11)

    return (
        _deg_out(degrees, rz),
        _deg_out(degrees, ry),
        _deg_out(degrees, rx),
    )


def quat_to_matrix(q: Quat) -> Tuple[Tuple[float, float, float], ...]:
    """可选：四元数 -> 3x3 旋转矩阵（行主序三元组）。"""
    w, x, y, z = quat_normalize(q)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("四元数 ↔ 欧拉角 (Z-Y-X)")
    # 默认与最小尺寸略大，适配高 DPI / 中文标签，避免底部说明被裁切
    root.minsize(560, 620)
    root.geometry("620x700")

    main = ttk.Frame(root, padding=14)
    main.pack(fill=tk.BOTH, expand=True)

    deg_var = tk.BooleanVar(value=True)

    intro = ttk.Label(
        main,
        text="欧拉角顺序：内旋 Z → Y' → X''（与 R = Rz·Ry·Rx 一致）",
        wraplength=560,
        justify=tk.LEFT,
    )
    intro.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    tip = ttk.Label(
        main,
        text="提示：在对应区域填好数值后点击按钮；四元数会自动归一化。",
        foreground="#555",
        wraplength=560,
        justify=tk.LEFT,
    )

    def _sync_wrap(_event: tk.Event | None = None) -> None:
        try:
            w = max(280, main.winfo_width() - 36)
        except tk.TclError:
            return
        intro.configure(wraplength=w)
        tip.configure(wraplength=w)

    main.bind("<Configure>", _sync_wrap)

    unit_frame = ttk.Frame(main)
    unit_frame.pack(anchor=tk.W, pady=(0, 10))
    ttk.Label(unit_frame, text="欧拉角单位：").pack(side=tk.LEFT)
    ttk.Radiobutton(unit_frame, text="度", variable=deg_var, value=True).pack(
        side=tk.LEFT, padx=(4, 0)
    )
    ttk.Radiobutton(unit_frame, text="弧度", variable=deg_var, value=False).pack(
        side=tk.LEFT, padx=(8, 0)
    )

    # --- 四元数 ---
    qf = ttk.LabelFrame(main, text="四元数 (w, x, y, z)", padding=8)
    qf.pack(fill=tk.X, pady=(0, 10))
    q_entries: list[ttk.Entry] = []
    ql = ("w", "x", "y", "z")
    for name in ql:
        row = ttk.Frame(qf)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=f"{name} =", width=6).pack(side=tk.LEFT)
        e = ttk.Entry(row, width=28)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        q_entries.append(e)

    def on_quat_to_euler() -> None:
        try:
            w = float(q_entries[0].get().strip())
            x = float(q_entries[1].get().strip())
            y = float(q_entries[2].get().strip())
            z = float(q_entries[3].get().strip())
            rz, ry, rx = quaternion_to_euler_zyx(
                (w, x, y, z), degrees=deg_var.get()
            )
            e_entries[0].delete(0, tk.END)
            e_entries[0].insert(0, f"{rz:.10g}")
            e_entries[1].delete(0, tk.END)
            e_entries[1].insert(0, f"{ry:.10g}")
            e_entries[2].delete(0, tk.END)
            e_entries[2].insert(0, f"{rx:.10g}")
        except ValueError as ex:
            messagebox.showerror("输入错误", str(ex))
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("转换失败", str(ex))

    ttk.Button(qf, text="转为欧拉角", command=on_quat_to_euler).pack(
        anchor=tk.E, pady=(6, 0)
    )

    # --- 欧拉角 ---
    ef = ttk.LabelFrame(main, text="欧拉角 Z-Y-X (rz, ry, rx)", padding=8)
    ef.pack(fill=tk.X, pady=(0, 10))
    e_entries: list[ttk.Entry] = []
    el = ("rz (绕Z)", "ry (绕Y')", "rx (绕X'')")
    for label in el:
        row = ttk.Frame(ef)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        e = ttk.Entry(row, width=28)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        e_entries.append(e)

    def on_euler_to_quat() -> None:
        try:
            rz = float(e_entries[0].get().strip())
            ry = float(e_entries[1].get().strip())
            rx = float(e_entries[2].get().strip())
            q = euler_zyx_to_quaternion(rz, ry, rx, degrees=deg_var.get())
            for i, v in enumerate(q):
                q_entries[i].delete(0, tk.END)
                q_entries[i].insert(0, f"{v:.10g}")
        except ValueError as ex:
            messagebox.showerror("输入错误", str(ex))
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("转换失败", str(ex))

    ttk.Button(ef, text="转为四元数", command=on_euler_to_quat).pack(
        anchor=tk.E, pady=(6, 0)
    )

    ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
    tip.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))

    root.after_idle(_sync_wrap)
    root.mainloop()


def _main_cli() -> None:
    p = argparse.ArgumentParser(description="四元数 <-> Z-Y-X 欧拉角")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("e2q", help="欧拉角 (rz ry rx) -> 四元数 (w x y z)")
    pe.add_argument("rz", type=float)
    pe.add_argument("ry", type=float)
    pe.add_argument("rx", type=float)
    pe.add_argument("--deg", action="store_true", help="输入为角度制")

    pq = sub.add_parser("q2e", help="四元数 (w x y z) -> 欧拉角 (rz ry rx)")
    pq.add_argument("w", type=float)
    pq.add_argument("x", type=float)
    pq.add_argument("y", type=float)
    pq.add_argument("z", type=float)
    pq.add_argument("--deg", action="store_true", help="输出为角度制")

    sub.add_parser("gui", help="打开图形界面")

    args = p.parse_args()
    if args.cmd == "gui":
        launch_gui()
        return
    if args.cmd == "e2q":
        q = euler_zyx_to_quaternion(args.rz, args.ry, args.rx, degrees=args.deg)
        print(f"w x y z = {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g}")
    else:
        rz, ry, rx = quaternion_to_euler_zyx(
            (args.w, args.x, args.y, args.z), degrees=args.deg
        )
        unit = "deg" if args.deg else "rad"
        print(f"rz ry rx ({unit}) = {rz:.12g} {ry:.12g} {rx:.12g}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        launch_gui()
    else:
        _main_cli()
