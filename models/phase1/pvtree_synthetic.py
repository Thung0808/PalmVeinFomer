from __future__ import annotations

import json
import logging
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("pvtree_synth")


@dataclass
class PVTreeSynthConfig:
    image_size: int = 224
    num_subjects: int = 8000
    samples_per_subject: int = 8
    synth_mode: str = "nir_fallback"  # pattern | nir_fallback
    volume_width: float = 70.0
    volume_depth: float = 14.0
    volume_height: float = 80.0
    volume_y_center: float = 40.0
    r_ori: float = 0.5
    ratioE: float = 0.065
    ratioQ: float = 0.5
    gamma: float = 3.0
    branch_points: int = 70
    kamyia_iters: int = 50
    branch_min_dist: float = 3.0
    trunk_case_probs: tuple[float, float, float, float] = (0.30, 0.30, 0.20, 0.20)
    trunk_x_jitter: float = 3.0
    trunk_y_jitter: float = 7.0
    trunk_z_jitter: float = 5.0
    finger_x_jitter: float = 1.5
    finger_y_jitter: float = 7.0
    finger_z_jitter: float = 2.0
    step_2d: float = 1.0
    tsm_rand_weight: float = 0.30
    view_angle_min_deg: float = -3.0
    view_angle_max_deg: float = 3.0
    depth_w_min: float = 0.95
    depth_w_max: float = 1.05
    vein_width_scale: float = 1.6
    pattern_blur_sigma_min: float = 0.6
    pattern_blur_sigma_max: float = 1.3
    augment_rotate_deg: float = 4.0
    augment_scale_min: float = 0.97
    augment_scale_max: float = 1.03
    augment_translate_px: float = 4.0
    augment_perspective: float = 0.05
    augment_crop_min: float = 0.92
    use_bezier_crease: bool = True
    bezier_main_width_min: float = 1.2
    bezier_main_width_max: float = 2.3
    bezier_secondary_min: int = 3
    bezier_secondary_max: int = 8
    bezier_secondary_width_min: float = 0.8
    bezier_secondary_width_max: float = 1.5
    bezier_secondary_len_min: float = 0.08
    bezier_secondary_len_max: float = 0.30
    nir_target_mean: float = 132.0
    nir_target_std: float = 18.0
    nir_target_mean_jitter: float = 7.0
    nir_target_std_jitter: float = 3.0
    nir_vein_strength_min: float = 18.0
    nir_vein_strength_max: float = 32.0
    nir_bg_mean_min: float = 126.0
    nir_bg_mean_max: float = 146.0
    nir_bg_radial_min: float = 2.5
    nir_bg_radial_max: float = 8.0
    nir_bg_linear_min: float = -5.0
    nir_bg_linear_max: float = 5.0
    nir_lowfreq_min: float = 1.0
    nir_lowfreq_max: float = 3.5
    nir_sensor_noise_min: float = 1.0
    nir_sensor_noise_max: float = 2.8
    nir_global_blur_min: float = 0.3
    nir_global_blur_max: float = 0.9
    palm_mask_soft_sigma: float = 4.0


@dataclass(eq=False)
class _Point:
    position: np.ndarray
    radius: float
    num_end: int = 0
    parent: "_Point | None" = None


@dataclass(eq=False)
class _Seg:
    point_in: _Point
    point_out: _Point


@dataclass
class _Style:
    bg_mean: float
    bg_radial: float
    bg_linear: float
    lowfreq_gain: float
    vein_strength: float
    noise_sigma: float
    global_blur: float
    target_mean: float
    target_std: float


_CASE_DEFS = {
    1: {
        "root_x": [33.0, 22.0, 18.0, 25.0, 30.0, 38.0, 46.0, 50.0, 45.0, 38.0],
        "root_z": [-3.0, 30.0, 42.0, 50.0, 55.0, 58.0, -3.0, 31.0, 45.0, 58.0],
        "root_num": [9, 9, 8, 6, 4, 2, 5, 5, 4, 2],
        "root_edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9)],
        "s_finger": [1, 2, 3, 4, 5, 8, 7],
        "branch_x": [5, 6, 15, 28, 47, 58, 70],
        "branch_z": [23, 55, 66, 73, 72, 53, 30],
        "branch_num": [1, 2, 2, 2, 2, 2, 1],
        "branch_to_s": [0, 1, 2, 3, 4, 5, 6],
        "finger_to_branch": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    },
    2: {
        "root_x": [30.0, 25.0, 20.0, 22.0, 33.0, 41.0, 51.0, 50.0, 47.0],
        "root_z": [-5.0, 24.0, 35.0, 52.0, 58.0, -3.0, 27.0, 39.0, 58.0],
        "root_num": [7, 7, 6, 4, 2, 5, 5, 4, 2],
        "root_edges": [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)],
        "s_finger": [1, 2, 3, 4, 8, 7, 6],
        "branch_x": [5, 6, 18, 34, 48, 60, 70],
        "branch_z": [13, 56, 68, 70, 70, 45, 30],
        "branch_num": [1, 2, 2, 2, 2, 2, 1],
        "branch_to_s": [0, 1, 2, 3, 4, 5, 6],
        "finger_to_branch": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    },
    3: {
        "root_x": [32.0, 20.0, 28.0, 36.0, 46.0, 60.0, 42.0, 45.0],
        "root_z": [-3.0, 40.0, 50.0, 57.0, 60.0, 65.0, -3.0, 25.0],
        "root_num": [10, 10, 8, 6, 4, 2, 2, 2],
        "root_edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)],
        "s_finger": [1, 2, 3, 4, 5, 7],
        "branch_x": [5, 16, 33, 50],
        "branch_z": [57, 64, 72, 72],
        "branch_num": [2, 2, 2, 2],
        "extra_branch_from_s": [4, 5],
        "branch_to_s": [0, 1, 2, 3, -1, -1],
        "finger_to_branch": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    },
    4: {
        "root_x": [30.0, 22.0, 42.0, 45.0, 40.0, 30.0, 42.0],
        "root_z": [-3.0, 30.0, -3.0, 25.0, 45.0, 50.0, 54.0],
        "root_num": [1, 1, 10, 10, 8, 4, 4],
        "root_edges": [(0, 1), (2, 3), (3, 4), (4, 5), (4, 6)],
        "s_finger": [1, 5, 6, 3],
        "branch_x": [0, 19, 20, 36, 50, 57],
        "branch_z": [38, 60, 70, 74, 72, 53],
        "branch_num": [0, 2, 2, 2, 2, 2],
        "branch_to_s": [0, 1, 1, 2, 2, 3],
        "finger_to_branch": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    },
}

_FINGER_COORDS = {
    1: ([0, 3, 7, 20, 25, 37, 41, 58, 65, 70], [52, 68, 71, 76, 80, 80, 80, 76, 71, 50]),
    2: ([0, 8, 13, 23, 30, 40, 47, 60, 65, 70], [58, 70, 72, 76, 80, 80, 80, 76, 72, 48]),
    3: ([0, 4, 9, 18, 26, 36, 43, 58, 65, 70, 53, 65], [56, 64, 66, 75, 77, 80, 80, 80, 72, 48, 42, 34]),
    4: ([0, 6, 10, 25, 30, 40, 48, 60, 66, 70], [56, 68, 72, 75, 80, 80, 80, 75, 70, 50]),
}

_B_AXIS = ((-0.75, -0.20, 0.20, 1.20), (2.60, 3.25, 1.25, 1.50), (2.20, 2.40, 1.45, 1.65))
_B_T = ((0.40, 0.60), (0.40, 0.60), (0.40, 0.60))
_B_S = ((-0.05, 0.40), (0.05, 0.30), (0.05, 0.30))


def _u(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.zeros_like(v, dtype=np.float32) if n < eps else (v / n).astype(np.float32)


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cpu() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def _mk_point(cfg: PVTreeSynthConfig, rng: np.random.Generator, x: float, z: float, n: int) -> _Point:
    p = np.array(
        [
            x + rng.uniform(-cfg.trunk_x_jitter, cfg.trunk_x_jitter),
            cfg.volume_y_center + rng.uniform(-cfg.trunk_y_jitter, cfg.trunk_y_jitter),
            z + rng.uniform(-cfg.trunk_z_jitter, cfg.trunk_z_jitter),
        ],
        dtype=np.float32,
    )
    return _Point(p, cfg.r_ori + n * cfg.ratioE, n, None)


def _refresh(cfg: PVTreeSynthConfig, segs: list[_Seg]) -> None:
    for s in segs:
        s.point_in.radius = cfg.r_ori + s.point_in.num_end * cfg.ratioE
        s.point_out.radius = cfg.r_ori + s.point_out.num_end * cfg.ratioE


def _choose_case(cfg: PVTreeSynthConfig, rng: np.random.Generator) -> int:
    p = np.clip(np.asarray(cfg.trunk_case_probs, dtype=np.float64), 0.0, None)
    p = np.full(4, 0.25, dtype=np.float64) if float(p.sum()) <= 0 else p / p.sum()
    return int(rng.choice(np.array([1, 2, 3, 4]), p=p))


def _finger_positions(cfg: PVTreeSynthConfig, rng: np.random.Generator, case_id: int) -> list[np.ndarray]:
    xs, zs = _FINGER_COORDS[case_id]
    out: list[np.ndarray] = []
    for x, z in zip(xs, zs):
        out.append(
            np.array(
                [
                    x + rng.uniform(-cfg.finger_x_jitter, cfg.finger_x_jitter),
                    cfg.volume_y_center + rng.uniform(-cfg.finger_y_jitter, cfg.finger_y_jitter),
                    z + rng.uniform(-cfg.finger_z_jitter, cfg.finger_z_jitter),
                ],
                dtype=np.float32,
            )
        )
    return out


def _build_trunk(cfg: PVTreeSynthConfig, rng: np.random.Generator, case_id: int) -> list[_Seg]:
    dct = _CASE_DEFS[case_id]
    roots = [_mk_point(cfg, rng, x, z, n) for x, z, n in zip(dct["root_x"], dct["root_z"], dct["root_num"])]
    segs: list[_Seg] = []
    for a, b in dct["root_edges"]:
        roots[b].parent = roots[a]
        segs.append(_Seg(roots[a], roots[b]))

    s_finger = [roots[i] for i in dct["s_finger"]]
    branches = [_mk_point(cfg, rng, x, z, n) for x, z, n in zip(dct["branch_x"], dct["branch_z"], dct["branch_num"])]
    for idx in dct.get("extra_branch_from_s", []):
        branches.append(s_finger[idx])

    finger_pos = _finger_positions(cfg, rng, case_id)
    for i, bidx in enumerate(dct["finger_to_branch"]):
        fp = _Point(finger_pos[i], cfg.r_ori, 0, branches[bidx])
        segs.append(_Seg(branches[bidx], fp))

    for bi, si in enumerate(dct["branch_to_s"]):
        if si < 0:
            continue
        branches[bi].parent = s_finger[si]
        segs.append(_Seg(s_finger[si], branches[bi]))

    _refresh(cfg, segs)
    return segs


def _point_seg_dist(p: np.ndarray, s: _Seg) -> float:
    a, b = s.point_in.position, s.point_out.position
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 1e-8:
        return _d(p, a)
    t = float(np.dot(p - a, ab) / den)
    t = max(0.0, min(1.0, t))
    return _d(p, a + t * ab)


def _nearest(cfg: PVTreeSynthConfig, p: np.ndarray, segs: list[_Seg]) -> _Seg | None:
    if not segs:
        return None
    best, best_d = None, float("inf")
    for s in segs:
        d = _point_seg_dist(p, s)
        if d >= cfg.branch_min_dist and d < best_d:
            best, best_d = s, d
    if best is not None:
        return best
    for s in segs:
        d = _point_seg_dist(p, s)
        if d < best_d:
            best, best_d = s, d
    return best


def _has_child(p: _Point, segs: list[_Seg]) -> bool:
    for s in segs:
        if s.point_in is p:
            return True
    return False


def _kamyia(cfg: PVTreeSynthConfig, p2: _Point, min_seg: _Seg, segs: list[_Seg]) -> None:
    p0, p1 = min_seg.point_in, min_seg.point_out
    r0, r1, r2 = max(p0.radius, 1e-6), max(p1.radius, 1e-6), max(p2.radius, 1e-6)
    f0 = max(r0**3, 1e-8)
    f1 = max(cfg.ratioQ * f0, 1e-8)
    f2 = max((1.0 - cfg.ratioQ) * f0, 1e-8)
    pbp = ((f0 * p0.position) + (f1 * p1.position) + (f2 * p2.position)) / (2.0 * f0)
    pb = _Point(pbp.astype(np.float32), cfg.r_ori, 0, None)
    l0, l1, l2 = _d(pb.position, p0.position), _d(pb.position, p1.position), _d(pb.position, p2.position)
    for _ in range(int(cfg.kamyia_iters)):
        R1, R2 = max(r1 * r1, 1e-8), max(r2 * r2, 1e-8)
        g = max(float(cfg.gamma), 1e-6)
        R0 = max(f0 * ((R1**g) / f1 + (R2**g) / f2), 1e-8) ** (1.0 / g)
        den = (R0 / max(l0, 1e-6)) + (R1 / max(l1, 1e-6)) + (R2 / max(l2, 1e-6))
        if den <= 1e-8:
            break
        pb.position = (
            p0.position * (R0 / max(l0, 1e-6)) + p1.position * (R1 / max(l1, 1e-6)) + p2.position * (R2 / max(l2, 1e-6))
        ) / den
        r0, r1, r2 = math.sqrt(max(R0, 1e-8)), math.sqrt(max(R1, 1e-8)), math.sqrt(max(R2, 1e-8))
        f0, f1, f2 = max(r0**3, 1e-8), max(r1**3, 1e-8), max(r2**3, 1e-8)
        l0, l1, l2 = _d(pb.position, p0.position), _d(pb.position, p1.position), _d(pb.position, p2.position)
        if min(l0, l1, l2) < 1.0:
            break
    p1.parent, p2.parent, pb.parent = pb, pb, p0
    pb.num_end = p1.num_end + 1 if _has_child(p1, segs) else 2
    pb.radius = cfg.r_ori + pb.num_end * cfg.ratioE
    segs.remove(min_seg)
    segs.extend([_Seg(p0, pb), _Seg(pb, p1), _Seg(pb, p2)])
    p = pb
    while p.parent is not None:
        p.parent.num_end += 1
        p.parent.radius = cfg.r_ori + p.parent.num_end * cfg.ratioE
        p = p.parent


def _grow(cfg: PVTreeSynthConfig, rng: np.random.Generator, segs: list[_Seg]) -> None:
    for _ in range(int(cfg.branch_points)):
        p = np.array(
            [
                rng.uniform(0.0, cfg.volume_width),
                rng.uniform(cfg.volume_y_center - cfg.volume_depth * 0.5, cfg.volume_y_center + cfg.volume_depth * 0.5),
                rng.uniform(0.0, cfg.volume_height),
            ],
            dtype=np.float32,
        )
        m = _nearest(cfg, p, segs)
        if m is None:
            continue
        _kamyia(cfg, _Point(p, cfg.r_ori, 0, None), m, segs)


def _collect_pts(segs: list[_Seg]) -> list[_Point]:
    seen, out = set(), []
    for s in segs:
        if s.point_in not in seen:
            seen.add(s.point_in)
            out.append(s.point_in)
        if s.point_out not in seen:
            seen.add(s.point_out)
            out.append(s.point_out)
    return out


def _rotate(cfg: PVTreeSynthConfig, segs: list[_Seg], ang: float) -> dict[_Point, np.ndarray]:
    c, s = math.cos(ang), math.sin(ang)
    ctr = np.array([cfg.volume_width * 0.5, cfg.volume_y_center, cfg.volume_height * 0.5], dtype=np.float32)
    out: dict[_Point, np.ndarray] = {}
    for p in _collect_pts(segs):
        v = p.position - ctr
        out[p] = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=np.float32) + ctr
    return out


def _to_px(cfg: PVTreeSynthConfig, p: np.ndarray) -> tuple[int, int]:
    n = cfg.image_size - 1
    x = int(round((p[0] / max(cfg.volume_width, 1e-6)) * n))
    y = int(round((1.0 - p[2] / max(cfg.volume_height, 1e-6)) * n))
    return int(np.clip(x, 0, n)), int(np.clip(y, 0, n))


def _traj(cfg: PVTreeSynthConfig, rng: np.random.Generator, ps: np.ndarray, pe: np.ndarray) -> list[np.ndarray]:
    pts, cur = [ps.copy()], ps.copy()
    for _ in range(512):
        if _d(cur, pe) <= cfg.step_2d:
            break
        d = _u(_u(pe - cur) + cfg.tsm_rand_weight * _u(rng.normal(size=3).astype(np.float32)))
        cur = cur + cfg.step_2d * d
        pts.append(cur.copy())
    pts.append(pe.copy())
    return pts


def _render_pattern(cfg: PVTreeSynthConfig, rng: np.random.Generator, segs: list[_Seg]) -> np.ndarray:
    size = cfg.image_size
    rot = _rotate(cfg, segs, math.radians(rng.uniform(cfg.view_angle_min_deg, cfg.view_angle_max_deg)))
    ys = [float(v[1]) for v in rot.values()]
    y0, y1 = min(ys), max(ys)
    wd = float(rng.uniform(cfg.depth_w_min, cfg.depth_w_max))
    img = np.full((size, size), 255.0, dtype=np.float32)
    gain = cfg.vein_width_scale * (size / max(cfg.volume_width, 1e-6))
    den = max(y1 - y0, 1e-6)
    for s in segs:
        curve = _traj(cfg, rng, rot[s.point_out], rot[s.point_in])
        t = max(1, int(round(max(s.point_out.radius, 0.4) * gain)))
        for a, b in zip(curve[:-1], curve[1:]):
            g = ((float(a[1]) - y0) * wd / den) * 255.0
            col = float(np.clip(22.0 + 0.78 * g, 0.0, 235.0))
            cv2.line(img, _to_px(cfg, a), _to_px(cfg, b), color=col, thickness=t, lineType=cv2.LINE_AA)
    sig = float(rng.uniform(cfg.pattern_blur_sigma_min, cfg.pattern_blur_sigma_max))
    return np.clip(cv2.GaussianBlur(img, (0, 0), sigmaX=sig, sigmaY=sig), 0, 255).astype(np.uint8)


def _edge(rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    off = min(lo, hi)
    t = float(rng.uniform(lo - off, hi - off) + off)
    t = t % 4.0
    if t <= 1.0:
        return np.array([t, 0.0], dtype=np.float32)
    if t <= 2.0:
        return np.array([1.0, t - 1.0], dtype=np.float32)
    if t <= 3.0:
        return np.array([3.0 - t, 1.0], dtype=np.float32)
    return np.array([0.0, 4.0 - t], dtype=np.float32)


def _ctrl(a: np.ndarray, b: np.ndarray, t: float, s: float) -> np.ndarray:
    c = a * t + b * (1.0 - t)
    v = _u(np.array([-(a - b)[1], (a - b)[0]], dtype=np.float32))
    return c + s * float(np.linalg.norm(a - b)) * v


def _bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    o = 1.0 - t
    return ((o * o) * p0[None, :] + (2.0 * o * t) * p1[None, :] + (t * t) * p2[None, :]).astype(np.float32)


def _draw_poly(img: np.ndarray, pts: np.ndarray, w: float) -> None:
    th = max(1, int(round(float(w))))
    for i in range(len(pts) - 1):
        a = (int(round(float(pts[i][0]))), int(round(float(pts[i][1]))))
        b = (int(round(float(pts[i + 1][0]))), int(round(float(pts[i + 1][1]))))
        cv2.line(img, a, b, color=0, thickness=th, lineType=cv2.LINE_AA)


def _crease(cfg: PVTreeSynthConfig, rng: np.random.Generator) -> np.ndarray:
    s = cfg.image_size
    img = np.full((s, s), 255, dtype=np.uint8)
    for i in range(3):
        h, t = _edge(rng, _B_AXIS[i][0], _B_AXIS[i][1]), _edge(rng, _B_AXIS[i][2], _B_AXIS[i][3])
        tv = float(rng.uniform(_B_T[i][0], _B_T[i][1]))
        sv = float(rng.uniform(_B_S[i][0], _B_S[i][1]))
        if i == 0:
            sv = -sv
        pts = _bezier(h * s, _ctrl(h, t, tv, sv) * s, t * s, 80)
        _draw_poly(img, pts, rng.uniform(cfg.bezier_main_width_min, cfg.bezier_main_width_max))
    n = int(rng.integers(cfg.bezier_secondary_min, cfg.bezier_secondary_max + 1))
    for _ in range(n):
        p0 = rng.uniform(0.0, s, size=2).astype(np.float32)
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        ln = float(rng.uniform(cfg.bezier_secondary_len_min, cfg.bezier_secondary_len_max) * s)
        p2 = p0 + np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * ln
        cv = _ctrl(p0, p2, float(rng.uniform(0.3, 0.7)), float(np.clip(rng.normal(0.0, 0.4), -0.6, 0.6)))
        _draw_poly(img, _bezier(p0, cv, p2, 50), rng.uniform(cfg.bezier_secondary_width_min, cfg.bezier_secondary_width_max))
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.6, sigmaY=0.6)
    return img


def _augment(cfg: PVTreeSynthConfig, rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    s = cfg.image_size
    c = (s * 0.5, s * 0.5)
    m = cv2.getRotationMatrix2D(
        c,
        float(rng.uniform(-cfg.augment_rotate_deg, cfg.augment_rotate_deg)),
        float(rng.uniform(cfg.augment_scale_min, cfg.augment_scale_max)),
    )
    m[0, 2] += float(rng.uniform(-cfg.augment_translate_px, cfg.augment_translate_px))
    m[1, 2] += float(rng.uniform(-cfg.augment_translate_px, cfg.augment_translate_px))
    out = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if cfg.augment_perspective > 0:
        mar = cfg.augment_perspective * s
        src = np.array([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], dtype=np.float32)
        dst = src + rng.uniform(-mar, mar, size=(4, 2)).astype(np.float32)
        out = cv2.warpPerspective(
            out,
            cv2.getPerspectiveTransform(src, dst),
            (s, s),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
    r = float(rng.uniform(cfg.augment_crop_min, 1.0))
    if r < 0.999:
        cs = max(16, int(round(s * r)))
        x0 = int(rng.integers(0, s - cs + 1))
        y0 = int(rng.integers(0, s - cs + 1))
        out = cv2.resize(out[y0 : y0 + cs, x0 : x0 + cs], (s, s), interpolation=cv2.INTER_LINEAR)
    return out


def _mask(cfg: PVTreeSynthConfig, rng: np.random.Generator) -> np.ndarray:
    s = cfg.image_size
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    xn = (xx - s * 0.5) / (s * 0.48)
    yn = (yy - s * 0.60) / (s * 0.52)
    base = np.clip(1.15 - np.sqrt(np.clip(xn * xn + yn * yn, 0.0, 4.0)), 0.0, 1.0)
    tc = float(s * (0.18 + rng.normal(0.0, 0.015)))
    tf = 1.0 / (1.0 + np.exp(-(yy - tc) / (s * 0.03)))
    m = base * tf
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigmaX=cfg.palm_mask_soft_sigma, sigmaY=cfg.palm_mask_soft_sigma), 0.0, 1.0)


def _match(img: np.ndarray, tm: float, ts: float) -> np.ndarray:
    x = img.astype(np.float32)
    m, s = float(x.mean()), float(x.std())
    s = 1.0 if s < 1e-6 else s
    return np.clip((x - m) * (ts / s) + tm, 0, 255).astype(np.uint8)


def _sample_style(cfg: PVTreeSynthConfig, rng: np.random.Generator) -> _Style:
    return _Style(
        float(rng.uniform(cfg.nir_bg_mean_min, cfg.nir_bg_mean_max)),
        float(rng.uniform(cfg.nir_bg_radial_min, cfg.nir_bg_radial_max)),
        float(rng.uniform(cfg.nir_bg_linear_min, cfg.nir_bg_linear_max)),
        float(rng.uniform(cfg.nir_lowfreq_min, cfg.nir_lowfreq_max)),
        float(rng.uniform(cfg.nir_vein_strength_min, cfg.nir_vein_strength_max)),
        float(rng.uniform(cfg.nir_sensor_noise_min, cfg.nir_sensor_noise_max)),
        float(rng.uniform(cfg.nir_global_blur_min, cfg.nir_global_blur_max)),
        float(rng.normal(cfg.nir_target_mean, cfg.nir_target_mean_jitter)),
        float(max(10.0, rng.normal(cfg.nir_target_std, cfg.nir_target_std_jitter))),
    )


def _render_nir(cfg: PVTreeSynthConfig, rng: np.random.Generator, pat: np.ndarray, st: _Style) -> np.ndarray:
    s = cfg.image_size
    edge = cv2.GaussianBlur(_mask(cfg, rng), (0, 0), sigmaX=cfg.palm_mask_soft_sigma, sigmaY=cfg.palm_mask_soft_sigma)
    vm = np.clip((255.0 - pat.astype(np.float32)) / 255.0, 0.0, 1.0)
    vm = cv2.GaussianBlur(vm, (0, 0), sigmaX=float(rng.uniform(0.8, 2.4)), sigmaY=float(rng.uniform(0.8, 2.4)))
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    xn = (xx - s * 0.5) / (s * 0.5)
    yn = (yy - s * 0.58) / (s * 0.5)
    rad = np.sqrt(np.clip(xn * xn + yn * yn, 0.0, 1.8))
    th = float(rng.uniform(0.0, 2.0 * math.pi))
    lin = xn * math.cos(th) + yn * math.sin(th)
    lf = cv2.resize(rng.normal(0.0, 1.0, size=(s // 8, s // 8)).astype(np.float32), (s, s), interpolation=cv2.INTER_CUBIC)
    bg = st.bg_mean - rad * (st.bg_radial + rng.normal(0.0, 0.7))
    bg = bg + lin * (st.bg_linear + rng.normal(0.0, 0.9))
    bg = bg + lf * (st.lowfreq_gain + rng.normal(0.0, 0.4))
    im = bg - vm * (st.vein_strength + float(rng.normal(0.0, 1.6)))
    outv = st.bg_mean - float(rng.uniform(1.0, 7.0))
    im = im * edge + outv * (1.0 - edge)
    im = im + rng.normal(0.0, st.noise_sigma, size=(s, s)).astype(np.float32)
    im = cv2.GaussianBlur(im, (0, 0), sigmaX=max(0.05, st.global_blur + float(rng.normal(0.0, 0.08))), sigmaY=max(0.05, st.global_blur + float(rng.normal(0.0, 0.08))))
    return _match(im, st.target_mean + float(rng.normal(0.0, 1.8)), max(8.0, st.target_std + float(rng.normal(0.0, 1.1))))


def _gen_subject(idx: int, root: Path, cfg: PVTreeSynthConfig, seed: int, overwrite: bool) -> tuple[int, int]:
    sd = root / f"subject_{idx + 1:05d}"
    exp = int(cfg.samples_per_subject)
    if sd.exists() and not overwrite and len(list(sd.glob("*.png"))) >= exp:
        return 0, 0
    if sd.exists() and overwrite:
        shutil.rmtree(sd)
    sd.mkdir(parents=True, exist_ok=True)
    sseed = int(seed + idx * 104_729)
    rs = np.random.default_rng(sseed)
    case_id = _choose_case(cfg, rs)
    segs = _build_trunk(cfg, rs, case_id)
    _grow(cfg, rs, segs)
    st = _sample_style(cfg, rs)
    n = 0
    for si in range(exp):
        rng = np.random.default_rng(sseed + si * 9_973)
        pat = _render_pattern(cfg, rng, segs)
        if cfg.use_bezier_crease:
            pat = np.minimum(pat, _crease(cfg, rng))
        pat = _augment(cfg, rng, pat)
        img = pat if cfg.synth_mode == "pattern" else _render_nir(cfg, rng, pat, st)
        p = sd / f"img_{si + 1:03d}.png"
        if not cv2.imwrite(str(p), img):
            raise RuntimeError(f"Failed to write synthetic image: {p}")
        n += 1
    return n, case_id


def generate_synthetic_dataset(
    output_root: str | Path,
    cfg: PVTreeSynthConfig,
    seed: int = 42,
    workers: int = 0,
    overwrite_root: bool = False,
    overwrite_subject: bool = False,
) -> dict[str, int | str | dict]:
    if cfg.synth_mode not in {"pattern", "nir_fallback"}:
        raise ValueError("--synth-mode must be one of: pattern, nir_fallback")
    output_root = Path(output_root)
    if overwrite_root and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    idxs = list(range(int(cfg.num_subjects)))
    total, hist = 0, {1: 0, 2: 0, 3: 0, 4: 0}
    wk = int(workers)
    if wk <= 0:
        wk = max(1, min(8, (_cpu() or 4) - 1))

    if wk == 1:
        for i in idxs:
            g, c = _gen_subject(i, output_root, cfg, seed, overwrite_subject)
            total += int(g)
            if c in hist:
                hist[c] += 1
    else:
        with ThreadPoolExecutor(max_workers=wk) as ex:
            futs = [ex.submit(_gen_subject, i, output_root, cfg, seed, overwrite_subject) for i in idxs]
            for f in as_completed(futs):
                g, c = f.result()
                total += int(g)
                if c in hist:
                    hist[c] += 1

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "seed": int(seed),
        "workers": int(wk),
        "num_subjects": int(cfg.num_subjects),
        "samples_per_subject": int(cfg.samples_per_subject),
        "generated_images": int(total),
        "case_histogram": {str(k): int(v) for k, v in hist.items()},
        "config": asdict(cfg),
    }
    with (output_root / "synthetic_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(
        "Synthetic generation done: subjects=%d samples_per_subject=%d generated_images=%d mode=%s",
        cfg.num_subjects,
        cfg.samples_per_subject,
        total,
        cfg.synth_mode,
    )
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    cfg = PVTreeSynthConfig(num_subjects=24, samples_per_subject=6, synth_mode="nir_fallback")
    m = generate_synthetic_dataset(Path("synthetic_preview"), cfg, seed=42, workers=4, overwrite_root=True, overwrite_subject=True)
    print(json.dumps(m, indent=2))
