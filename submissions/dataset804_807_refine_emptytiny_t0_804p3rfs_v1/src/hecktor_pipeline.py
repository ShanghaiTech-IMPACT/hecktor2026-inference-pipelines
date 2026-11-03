from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from tqdm import tqdm

os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache


CASE_ID = "hecktor_case"
SUBMISSION_VERSION = "dataset804_807_refine_emptytiny_t0_804p3rfs_v1"
FORBIDDEN_EHR_FIELDS = {"M-stage"}
SOFT_THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
SOFT_LITE_COMPONENT_THRESHOLDS = {
    "soft_gtvp": (0.05,),
    "soft_gtvn": (0.02, 0.05, 0.1, 0.2),
    "soft_fg": (0.02,),
}
SOFT_MIN_POSITIVE = 1e-6


def debug_log(message: str) -> None:
    if os.environ.get("HECKTOR_DEBUG", "").strip() == "1":
        print(f"[HECKTOR_DEBUG] {message}", flush=True)


def env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def env_folds(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    folds: list[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            folds.append(int(item))
        except ValueError:
            return default
    return tuple(folds) if folds else default


@dataclass(frozen=True)
class HnRoiConfig:
    ct_body_threshold_hu: float = -650.0
    detection_spacing_mm: float = 4.0
    xy_margin_mm: float = 35.0
    superior_margin_mm: float = 10.0
    brain_inferior_extension_mm: float = 320.0
    fallback_inferior_extension_mm: float = 420.0
    brain_threshold_ratios: tuple[float, ...] = (0.35, 0.30, 0.25)
    brain_suv_floor: float = 3.0
    brain_suv_cap: float = 8.0
    brain_min_volume_ml: float = 40.0
    brain_max_volume_ml: float = 2000.0
    brain_min_z_thickness_mm: float = 20.0
    brain_max_z_thickness_mm: float = 190.0
    brain_min_xy_extent_mm: float = 30.0
    brain_upper_body_start_fraction: float = 0.55


@dataclass(frozen=True)
class HeadEndQcConfig:
    ct_body_threshold_hu: float = -650.0
    pet_brain_threshold_ratios: tuple[float, ...] = (0.35, 0.30, 0.25)
    pet_brain_suv_floor: float = 3.0
    pet_brain_suv_cap: float = 8.0
    pet_brain_min_volume_ml: float = 40.0
    pet_brain_max_volume_ml: float = 2000.0
    pet_brain_min_z_extent_mm: float = 20.0
    pet_brain_max_z_extent_mm: float = 190.0
    pet_brain_min_xy_extent_mm: float = 30.0
    pet_side_fraction: float = 0.35
    ct_very_high_threshold_hu: float = 1200.0
    ct_component_min_volume_mm3: float = 20.0
    ct_component_max_z_extent_mm: float = 120.0
    ct_band_extent_mm: float = 150.0
    ct_band_max_fraction: float = 0.30
    ct_dominant_fraction: float = 0.60
    ct_dominant_ratio: float = 2.0
    ct_median_side_fraction: float = 0.35


@dataclass(frozen=True)
class WideHnRoiConfig:
    max_z_extent_mm: float = 500.0
    preserve_head_side: str = "end"
    head_end_qc: HeadEndQcConfig = HeadEndQcConfig()


@dataclass(frozen=True)
class LocalRoiConfig:
    target_shape_zyx: tuple[int, int, int] = (64, 128, 128)
    base_channels: int = 16
    downsample_levels: int = 4
    threshold_candidates: tuple[float, ...] = (
        0.50,
        0.20,
        0.02,
        0.002,
        0.0001,
    )
    xy_margin_mm: float = 55.0
    z_margin_mm: float = 70.0
    max_z_extent_mm: float = 340.0
    min_xy_size_mm: float = 180.0
    min_z_size_mm: float = 96.0
    fallback_min_roi_fraction: float = 0.015
    fallback_min_z_mm: float = 64.0
    component_min_voxels: int = 4
    component_min_relative_size: float = 0.02
    component_keep_top_k: int = 3
    component_keep_max_prob: float = 0.50
    component_far_gap_mm: float = 60.0
    qc_min_roi_fraction: float = 0.02
    qc_max_roi_fraction: float = 0.60
    qc_min_z_mm: float = 96.0
    qc_max_z_mm: float = 340.0
    qc_min_prob_max: float = 0.02
    qc_max_components_after: int = 30
    qc_low_conf_prob_max: float = 0.05
    qc_low_conf_min_roi_fraction: float = 0.10


@dataclass(frozen=True)
class RefinementRoiConfig:
    margin_xyz_mm: tuple[float, float, float] = (90.0, 90.0, 110.0)
    min_extent_xyz_mm: tuple[float, float, float] = (200.0, 200.0, 240.0)


@dataclass
class PreparedCase:
    case_id: str
    ct_path: Path
    pet_path: Path
    output_dir: Path
    ehr: dict[str, Any]
    case_start: float
    timing: dict[str, float]
    original_ct: sitk.Image
    ct_lps: sitk.Image
    hn_bbox: tuple[int, int, int, int, int, int]
    hn_info: dict[str, Any]
    hn_ct: sitk.Image
    hn_pet: sitk.Image
    tinyunet_input_ncdhw: np.ndarray


@dataclass
class LocalizedCase:
    prepared: PreparedCase
    local_bbox_hn: tuple[int, int, int, int, int, int]
    local_info: dict[str, Any]
    local_ct: sitk.Image
    local_pet: sitk.Image
    nnunet_data: np.ndarray
    nnunet_properties: dict[str, Any]
    previous_stage_seg: np.ndarray | None = None


@dataclass
class SegmentedCase:
    localized: LocalizedCase
    fine_seg_local: sitk.Image
    fine_prob_local: np.ndarray | None = None
    cascade_rfs_features: dict[str, float] | None = None


@dataclass(frozen=True)
class OfficialCaseSpec:
    case_id: str
    ct_path: Path
    pet_path: Path
    ehr: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True)
class BatchRuntimeConfig:
    mode: str = "sequential"
    cpu_workers: int = 1
    local_roi_workers: int = 1
    post_workers: int = 0
    prefetch: int = 1
    coarse_batch_size: int = 1

    @classmethod
    def from_env(cls, case_count: int) -> "BatchRuntimeConfig":
        default_mode = "threaded" if case_count > 1 else "sequential"
        mode = os.environ.get("HECKTOR_BATCH_MODE", "auto").strip().lower()
        if mode not in {"auto", "threaded", "sequential", "off"}:
            mode = "auto"
        if mode == "off":
            mode = "sequential"
        if mode == "auto":
            mode = default_mode
        if mode == "sequential":
            return cls()

        # Tuned for ml.g4dn.2xlarge: 8 vCPUs, 32 GiB RAM, 1x T4 16 GiB.
        cpu_workers = env_int("HECKTOR_CPU_WORKERS", 3, minimum=1)
        local_roi_workers = env_int("HECKTOR_LOCALROI_WORKERS", 2, minimum=1)
        post_workers = env_int("HECKTOR_POST_WORKERS", 1, minimum=0)
        prefetch_default = min(4, max(2, cpu_workers + 1))
        prefetch = env_int("HECKTOR_PREFETCH", prefetch_default, minimum=1)
        coarse_batch_size = env_int("HECKTOR_COARSE_BATCH_SIZE", 2, minimum=1)
        return cls(
            mode=mode,
            cpu_workers=cpu_workers,
            local_roi_workers=local_roi_workers,
            post_workers=post_workers,
            prefetch=prefetch,
            coarse_batch_size=coarse_batch_size,
        )


class BatchedTileNNUNetPredictor(nnUNetPredictor):
    """nnU-Net v2 predictor with optional batched sliding-window tile forwards."""

    def __init__(self, *args: Any, tile_batch_size: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tile_batch_size = max(1, int(tile_batch_size))

    def _predict_tile_batch_with_fallback(self, workon: torch.Tensor) -> torch.Tensor:
        try:
            return self._internal_maybe_mirror_and_predict(workon)
        except RuntimeError:
            if workon.shape[0] <= 1:
                raise
            empty_cache(self.device)
            midpoint = workon.shape[0] // 2
            left = self._predict_tile_batch_with_fallback(workon[:midpoint])
            right = self._predict_tile_batch_with_fallback(workon[midpoint:])
            return torch.cat([left, right], dim=0)

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers,
        do_on_device: bool = True,
    ):
        predicted_logits = n_predictions = predictions = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device("cpu")

        try:
            empty_cache(self.device)

            if self.verbose:
                print(f"move image to device {results_device}")
            data = data.to(results_device)

            if self.verbose:
                print(f"preallocating results arrays on device {results_device}")
            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.configuration_manager.patch_size),
                    sigma_scale=1.0 / 8,
                    value_scaling_factor=10,
                    device=results_device,
                )
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(
                    f"running batched prediction: {len(slicers)} steps, "
                    f"tile_batch_size={self.tile_batch_size}"
                )

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                for start in range(0, len(slicers), self.tile_batch_size):
                    batch_slicers = slicers[start : start + self.tile_batch_size]
                    workon = torch.cat(
                        [
                            torch.clone(data[s][None], memory_format=torch.contiguous_format)
                            for s in batch_slicers
                        ],
                        dim=0,
                    ).to(self.device)
                    predictions = self._predict_tile_batch_with_fallback(workon).to(results_device)

                    if self.use_gaussian:
                        predictions *= gaussian

                    for prediction, sl in zip(predictions, batch_slicers, strict=False):
                        predicted_logits[sl] += prediction
                        n_predictions[sl[1:]] += gaussian
                    pbar.update(len(batch_slicers))

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError(
                    "Encountered inf in predicted array. Aborting... If this problem persists, "
                    "reduce value_scaling_factor in compute_gaussian or increase the dtype of "
                    "predicted_logits to fp32"
                )
        except Exception as e:
            del predicted_logits, n_predictions, predictions, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = max(group for group in (4, 3, 2, 1) if out_channels % group == 0)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet3D(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 12) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.down1 = nn.Conv3d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(c * 2, c * 2)
        self.down2 = nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1)
        self.bridge = ConvBlock(c * 4, c * 4)
        self.up2 = nn.Conv3d(c * 4 + c * 2, c * 2, 1)
        self.dec2 = ConvBlock(c * 2, c * 2)
        self.up1 = nn.Conv3d(c * 2 + c, c, 1)
        self.dec1 = ConvBlock(c, c)
        self.out = nn.Conv3d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bridge(self.down2(e2))
        u2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(self.up2(torch.cat([u2, e2], dim=1)))
        u1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(self.up1(torch.cat([u1, e1], dim=1)))
        return self.out(d1)


class DeepTinyUNet3D(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 16, downsample_levels: int = 4) -> None:
        super().__init__()
        if downsample_levels < 3:
            raise ValueError("DeepTinyUNet3D is intended for at least 3 downsample levels.")
        if downsample_levels > 5:
            raise ValueError("downsample_levels > 5 is too memory-heavy for this localizer.")
        channels = [base_channels * (2**i) for i in range(downsample_levels + 1)]
        self.downsample_levels = downsample_levels
        self.enc0 = ConvBlock(in_channels, channels[0])
        self.downs = nn.ModuleList(
            [nn.Conv3d(channels[i], channels[i + 1], 3, stride=2, padding=1) for i in range(downsample_levels)]
        )
        self.encs = nn.ModuleList([ConvBlock(channels[i], channels[i]) for i in range(1, downsample_levels)])
        self.bridge = ConvBlock(channels[-1], channels[-1])
        self.reduces = nn.ModuleList(
            [nn.Conv3d(channels[i + 1] + channels[i], channels[i], 1) for i in reversed(range(downsample_levels))]
        )
        self.decs = nn.ModuleList([ConvBlock(channels[i], channels[i]) for i in reversed(range(downsample_levels))])
        self.out = nn.Conv3d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = self.enc0(x)
        skips.append(x)
        for level, enc in enumerate(self.encs, start=1):
            x = enc(self.downs[level - 1](x))
            skips.append(x)
        x = self.bridge(self.downs[-1](x))
        for reduce, dec, skip in zip(self.reduces, self.decs, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = dec(reduce(torch.cat([x, skip], dim=1)))
        return self.out(x)


def make_tinyunet_model(base_channels: int, downsample_levels: int) -> nn.Module:
    if downsample_levels == 2:
        return TinyUNet3D(base_channels=base_channels)
    return DeepTinyUNet3D(base_channels=base_channels, downsample_levels=downsample_levels)


def orient_lps(image: sitk.Image) -> sitk.Image:
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    if orientation == "LPS":
        return image
    return sitk.DICOMOrient(image, "LPS")


def same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and all(abs(x - y) < 1e-4 for x, y in zip(a.GetSpacing(), b.GetSpacing()))
        and all(abs(x - y) < 1e-3 for x, y in zip(a.GetOrigin(), b.GetOrigin()))
        and all(abs(x - y) < 1e-5 for x, y in zip(a.GetDirection(), b.GetDirection()))
    )


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float,
    pixel_id: int | None = None,
) -> sitk.Image:
    out_pixel_id = moving.GetPixelID() if pixel_id is None else pixel_id
    if same_grid(moving, reference):
        if moving.GetPixelID() == out_pixel_id:
            return moving
        return sitk.Cast(moving, out_pixel_id)
    return sitk.Resample(moving, reference, sitk.Transform(), interpolator, default_value, out_pixel_id)


def make_valid_mask_like(image: sitk.Image) -> sitk.Image:
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    return mask + 1


def make_detection_reference(ct: sitk.Image, spacing_mm: float) -> sitk.Image:
    original_size = ct.GetSize()
    original_spacing = ct.GetSpacing()
    target_spacing = tuple(max(float(s), float(spacing_mm)) for s in original_spacing)
    target_size = tuple(
        max(1, int(math.ceil(original_size[i] * original_spacing[i] / target_spacing[i]))) for i in range(3)
    )
    reference = sitk.Image(target_size, sitk.sitkFloat32)
    reference.SetSpacing(target_spacing)
    reference.SetOrigin(ct.GetOrigin())
    reference.SetDirection(ct.GetDirection())
    return reference


def bbox_from_mask_zyx(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return mask.astype(bool)
    labels, num = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def make_body_mask(ct_det_arr: np.ndarray, cfg: HnRoiConfig) -> tuple[np.ndarray, bool]:
    body = np.isfinite(ct_det_arr) & (ct_det_arr > cfg.ct_body_threshold_hu)
    if not bool(body.any()):
        return body.astype(bool), False
    body = largest_component(body)
    body = ndi.binary_closing(body, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    body = ndi.binary_fill_holes(body)
    body = largest_component(body)
    return body.astype(bool), bool(body.any())


def clamp_bbox_zyx(
    bbox: tuple[int, int, int, int, int, int],
    shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    z0 = max(0, min(shape_zyx[0] - 1, z0))
    y0 = max(0, min(shape_zyx[1] - 1, y0))
    x0 = max(0, min(shape_zyx[2] - 1, x0))
    z1 = max(z0 + 1, min(shape_zyx[0], z1))
    y1 = max(y0 + 1, min(shape_zyx[1], y1))
    x1 = max(x0 + 1, min(shape_zyx[2], x1))
    return z0, z1, y0, y1, x0, x1


def map_detection_bbox_to_ct_bbox_xyz(
    bbox_zyx: tuple[int, int, int, int, int, int],
    det_ref: sitk.Image,
    ct: sitk.Image,
    extra_voxel_margin: int = 1,
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox_zyx
    p0 = det_ref.TransformContinuousIndexToPhysicalPoint((float(x0), float(y0), float(z0)))
    p1 = det_ref.TransformContinuousIndexToPhysicalPoint((float(x1), float(y1), float(z1)))
    c0 = ct.TransformPhysicalPointToContinuousIndex(p0)
    c1 = ct.TransformPhysicalPointToContinuousIndex(p1)
    lows = [math.floor(min(c0[i], c1[i])) - extra_voxel_margin for i in range(3)]
    highs = [math.ceil(max(c0[i], c1[i])) + extra_voxel_margin for i in range(3)]
    size = ct.GetSize()
    x0_ct = max(0, min(size[0] - 1, lows[0]))
    y0_ct = max(0, min(size[1] - 1, lows[1]))
    z0_ct = max(0, min(size[2] - 1, lows[2]))
    x1_ct = max(x0_ct + 1, min(size[0], highs[0]))
    y1_ct = max(y0_ct + 1, min(size[1], highs[1]))
    z1_ct = max(z0_ct + 1, min(size[2], highs[2]))
    return x0_ct, x1_ct, y0_ct, y1_ct, z0_ct, z1_ct


def crop_image_xyz(image: sitk.Image, bbox_xyz: tuple[int, int, int, int, int, int]) -> sitk.Image:
    x0, x1, y0, y1, z0, z1 = bbox_xyz
    return sitk.RegionOfInterest(image, size=[x1 - x0, y1 - y0, z1 - z0], index=[x0, y0, z0])


def paste_prediction_to_grid(
    prediction_crop: sitk.Image,
    reference: sitk.Image,
    bbox_xyz: tuple[int, int, int, int, int, int],
) -> sitk.Image:
    x0, x1, y0, y1, z0, z1 = bbox_xyz
    expected_size = (x1 - x0, y1 - y0, z1 - z0)
    prediction_crop = sitk.Cast(prediction_crop, sitk.sitkUInt8)
    if prediction_crop.GetSize() != expected_size:
        crop_ref = crop_image_xyz(reference, bbox_xyz)
        prediction_crop = resample_to_reference(prediction_crop, crop_ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    out = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    out.CopyInformation(reference)
    return sitk.Paste(out, prediction_crop, prediction_crop.GetSize(), [0, 0, 0], [int(x0), int(y0), int(z0)])


def find_brain_anchor(
    pet_det_arr: np.ndarray,
    body_mask: np.ndarray,
    pet_valid_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    cfg: HnRoiConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    pet = np.nan_to_num(pet_det_arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pet[pet < 0] = 0
    body_bbox = bbox_from_mask_zyx(body_mask)
    if body_bbox is None:
        return None, {"pet_positive_voxels": 0, "pet_p995": 0.0}

    z0_body, z1_body, _, _, _, _ = body_bbox
    body_depth = max(1, z1_body - z0_body)
    upper_start = z0_body + int(round(body_depth * cfg.brain_upper_body_start_fraction))
    z_grid = np.arange(pet.shape[0])[:, None, None]
    valid = body_mask & pet_valid_mask & (z_grid >= upper_start) & np.isfinite(pet) & (pet > 0)
    values = pet[valid]
    if values.size == 0:
        return None, {"pet_positive_voxels": 0, "pet_p995": 0.0}

    pet_p995 = float(np.percentile(values, 99.5))
    smoothed = ndi.gaussian_filter(pet, sigma=1.0)
    voxel_volume_ml = float(np.prod(spacing_xyz) / 1000.0)
    candidates: list[dict[str, Any]] = []

    for ratio in cfg.brain_threshold_ratios:
        threshold = max(cfg.brain_suv_floor, min(cfg.brain_suv_cap, ratio * pet_p995))
        candidate_mask = (smoothed >= threshold) & valid
        labels, num = ndi.label(candidate_mask, structure=np.ones((3, 3, 3), dtype=bool))
        if num == 0:
            continue
        for label_id, slices in enumerate(ndi.find_objects(labels), start=1):
            if slices is None:
                continue
            comp_view = labels[slices] == label_id
            voxel_count = int(comp_view.sum())
            if voxel_count == 0:
                continue
            z_slice, y_slice, x_slice = slices
            z0, z1 = z_slice.start, z_slice.stop
            y0, y1 = y_slice.start, y_slice.stop
            x0, x1 = x_slice.start, x_slice.stop
            volume_ml = voxel_count * voxel_volume_ml
            z_thickness_mm = (z1 - z0) * spacing_xyz[2]
            x_extent_mm = (x1 - x0) * spacing_xyz[0]
            y_extent_mm = (y1 - y0) * spacing_xyz[1]
            z_center = 0.5 * (z0 + z1)
            z_center_rel = (z_center - z0_body) / body_depth
            if not (cfg.brain_min_volume_ml <= volume_ml <= cfg.brain_max_volume_ml):
                continue
            if not (cfg.brain_min_z_thickness_mm <= z_thickness_mm <= cfg.brain_max_z_thickness_mm):
                continue
            if min(x_extent_mm, y_extent_mm) < cfg.brain_min_xy_extent_mm:
                continue
            if z_center_rel < cfg.brain_upper_body_start_fraction:
                continue
            comp_values = pet[slices][comp_view]
            mean_pet = float(comp_values.mean()) if comp_values.size else 0.0
            score = (
                2.0 * z_center_rel
                + 0.8 * min(volume_ml / 350.0, 1.0)
                + 0.3 * min(z_thickness_mm / 100.0, 1.0)
                + 0.2 * min(mean_pet / max(threshold, 1e-3), 2.0)
            )
            candidates.append(
                {
                    "threshold_ratio": float(ratio),
                    "threshold": float(threshold),
                    "bbox_zyx": [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
                    "volume_ml": float(volume_ml),
                    "z_thickness_mm": float(z_thickness_mm),
                    "x_extent_mm": float(x_extent_mm),
                    "y_extent_mm": float(y_extent_mm),
                    "z_center_rel": float(z_center_rel),
                    "score": float(score),
                }
            )
        if candidates:
            break

    stats = {"pet_positive_voxels": int(values.size), "pet_p995": pet_p995, "candidate_count": len(candidates)}
    if not candidates:
        return None, stats
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0], stats


def compute_hn_roi(
    ct: sitk.Image,
    pet: sitk.Image,
    cfg: HnRoiConfig,
) -> tuple[tuple[int, int, int, int, int, int], dict[str, Any]]:
    det_ref = make_detection_reference(ct, cfg.detection_spacing_mm)
    ct_det = resample_to_reference(ct, det_ref, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    pet_det = resample_to_reference(pet, det_ref, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    ct_det_arr = sitk.GetArrayFromImage(ct_det)
    pet_det_arr = sitk.GetArrayFromImage(pet_det)
    if same_grid(ct, pet):
        pet_valid_arr = np.ones_like(pet_det_arr, dtype=bool)
        pet_valid_source = "ct_pet_same_grid"
    else:
        pet_valid = resample_to_reference(make_valid_mask_like(pet), det_ref, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
        pet_valid_arr = sitk.GetArrayFromImage(pet_valid) > 0
        pet_valid_source = "resampled_original_pet_fov"
    body_mask, body_found = make_body_mask(ct_det_arr, cfg)
    body_bbox = bbox_from_mask_zyx(body_mask)
    shape_zyx = ct_det_arr.shape
    spacing_xyz = det_ref.GetSpacing()
    qc_flags: list[str] = []

    if not body_found or body_bbox is None:
        qc_flags.append("body_foreground_failed")
        full_bbox = (0, shape_zyx[0], 0, shape_zyx[1], 0, shape_zyx[2])
        return map_detection_bbox_to_ct_bbox_xyz(full_bbox, det_ref, ct), {
            "roi_method": "full_fov_body_failed",
            "roi_confidence": 0.0,
            "qc_flags": qc_flags,
        }

    brain, pet_stats = find_brain_anchor(pet_det_arr, body_mask, pet_valid_arr, spacing_xyz, cfg)
    z0_body, z1_body, _, _, _, _ = body_bbox
    det_z_spacing = spacing_xyz[2]
    superior_pad = int(math.ceil(cfg.superior_margin_mm / det_z_spacing))
    if brain is not None:
        brain_z0 = int(brain["bbox_zyx"][0])
        z0 = int(math.floor(brain_z0 - cfg.brain_inferior_extension_mm / det_z_spacing))
        z1 = int(math.ceil(z1_body + superior_pad))
        roi_method = "pet_brain_anchor"
        roi_confidence = float(brain["score"])
    else:
        z0 = int(math.floor(z1_body - cfg.fallback_inferior_extension_mm / det_z_spacing))
        z1 = int(math.ceil(z1_body + superior_pad))
        roi_method = "ct_body_superior_fallback"
        roi_confidence = 0.25
        qc_flags.append("pet_brain_anchor_failed")

    z0 = max(0, z0)
    z1 = min(shape_zyx[0], max(z0 + 1, z1))
    slab_bbox = bbox_from_mask_zyx(body_mask[z0:z1])
    if slab_bbox is None:
        y0, y1, x0, x1 = 0, shape_zyx[1], 0, shape_zyx[2]
        qc_flags.append("body_slab_failed")
    else:
        _, _, y0_s, y1_s, x0_s, x1_s = slab_bbox
        y0 = y0_s - int(math.ceil(cfg.xy_margin_mm / spacing_xyz[1]))
        y1 = y1_s + int(math.ceil(cfg.xy_margin_mm / spacing_xyz[1]))
        x0 = x0_s - int(math.ceil(cfg.xy_margin_mm / spacing_xyz[0]))
        x1 = x1_s + int(math.ceil(cfg.xy_margin_mm / spacing_xyz[0]))

    det_bbox_zyx = clamp_bbox_zyx((z0, z1, y0, y1, x0, x1), shape_zyx)
    bbox_xyz = map_detection_bbox_to_ct_bbox_xyz(det_bbox_zyx, det_ref, ct)
    pet_valid_fraction = float(pet_valid_arr[body_mask].mean()) if bool(body_mask.any()) else 0.0
    if pet_valid_fraction < 0.20:
        qc_flags.append("low_pet_valid_fraction")

    return bbox_xyz, {
        "roi_method": roi_method,
        "roi_confidence": roi_confidence,
        "det_bbox_zyx": det_bbox_zyx,
        "body_found": bool(body_found),
        "body_bbox_zyx": body_bbox,
        "pet_valid_fraction_det_body": pet_valid_fraction,
        "pet_valid_source": pet_valid_source,
        "brain_anchor": brain,
        "pet_stats": pet_stats,
        "qc_flags": qc_flags,
    }


def qc_side_from_z_norm(z_norm: float, side_fraction: float) -> str:
    if z_norm <= side_fraction:
        return "start"
    if z_norm >= 1.0 - side_fraction:
        return "end"
    return "unknown"


def qc_connected_component_records(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> list[dict[str, Any]]:
    labels, num = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if num == 0:
        return []
    objects = ndi.find_objects(labels)
    voxel_volume_mm3 = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    records: list[dict[str, Any]] = []
    for label_idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        z_slc, y_slc, x_slc = slc
        local = labels[slc] == label_idx
        voxel_count = int(local.sum())
        if voxel_count == 0:
            continue
        z_idx, y_idx, x_idx = np.nonzero(local)
        records.append(
            {
                "label": int(label_idx),
                "voxel_count": voxel_count,
                "volume_mm3": float(voxel_count * voxel_volume_mm3),
                "bbox_z0": int(z_slc.start),
                "bbox_z1": int(z_slc.stop),
                "bbox_y0": int(y_slc.start),
                "bbox_y1": int(y_slc.stop),
                "bbox_x0": int(x_slc.start),
                "bbox_x1": int(x_slc.stop),
                "centroid_z": float((z_idx + z_slc.start).mean()),
                "centroid_y": float((y_idx + y_slc.start).mean()),
                "centroid_x": float((x_idx + x_slc.start).mean()),
            }
        )
    return records


def make_head_end_qc_body_mask(ct_arr: np.ndarray, cfg: HeadEndQcConfig) -> np.ndarray:
    body = np.isfinite(ct_arr) & (ct_arr > cfg.ct_body_threshold_hu)
    body = largest_component(body)
    if not bool(body.any()):
        return body.astype(bool)
    body = ndi.binary_closing(body, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    body = ndi.binary_fill_holes(body)
    return largest_component(body).astype(bool)


def find_pet_head_end_anchor(
    pet_arr: np.ndarray,
    body_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    cfg: HeadEndQcConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    pet = np.nan_to_num(pet_arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pet[pet < 0] = 0
    valid = body_mask & np.isfinite(pet) & (pet > 0)
    values = pet[valid]
    diagnostics: dict[str, Any] = {
        "pet_positive_voxels": int(values.size),
        "pet_p995": 0.0,
        "pet_threshold": 0.0,
        "pet_threshold_ratio": 0.0,
        "pet_candidates": 0,
    }
    if values.size == 0:
        return None, diagnostics

    pet_smooth = ndi.gaussian_filter(pet, sigma=1)
    p995 = float(np.percentile(pet_smooth[valid], 99.5))
    diagnostics["pet_p995"] = p995
    voxel_volume_ml = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]) / 1000.0
    candidates: list[dict[str, Any]] = []
    for ratio in cfg.pet_brain_threshold_ratios:
        threshold = max(cfg.pet_brain_suv_floor, min(cfg.pet_brain_suv_cap, float(ratio) * p995))
        mask = valid & (pet_smooth >= threshold)
        labels, num = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        if num == 0:
            continue
        for label_idx, slc in enumerate(ndi.find_objects(labels), start=1):
            if slc is None:
                continue
            z_slc, y_slc, x_slc = slc
            local = labels[slc] == label_idx
            voxels = int(local.sum())
            if voxels == 0:
                continue
            volume_ml = float(voxels * voxel_volume_ml)
            z_extent_mm = float((z_slc.stop - z_slc.start) * spacing_xyz[2])
            y_extent_mm = float((y_slc.stop - y_slc.start) * spacing_xyz[1])
            x_extent_mm = float((x_slc.stop - x_slc.start) * spacing_xyz[0])
            if not (cfg.pet_brain_min_volume_ml <= volume_ml <= cfg.pet_brain_max_volume_ml):
                continue
            if not (cfg.pet_brain_min_z_extent_mm <= z_extent_mm <= cfg.pet_brain_max_z_extent_mm):
                continue
            if max(x_extent_mm, y_extent_mm) < cfg.pet_brain_min_xy_extent_mm:
                continue
            z_idx, y_idx, x_idx = np.nonzero(local)
            z_global = z_idx + z_slc.start
            mean_pet = float(pet_smooth[slc][local].mean())
            candidates.append(
                {
                    "threshold_ratio": float(ratio),
                    "threshold": float(threshold),
                    "label": int(label_idx),
                    "volume_ml": volume_ml,
                    "z_extent_mm": z_extent_mm,
                    "x_extent_mm": x_extent_mm,
                    "y_extent_mm": y_extent_mm,
                    "centroid_z": float(z_global.mean()),
                    "centroid_y": float((y_idx + y_slc.start).mean()),
                    "centroid_x": float((x_idx + x_slc.start).mean()),
                    "mean_pet": mean_pet,
                    "score": float(volume_ml * mean_pet),
                }
            )
        if candidates:
            diagnostics["pet_threshold"] = float(threshold)
            diagnostics["pet_threshold_ratio"] = float(ratio)
            break

    diagnostics["pet_candidates"] = len(candidates)
    if not candidates:
        return None, diagnostics
    best = max(candidates, key=lambda r: r["score"])
    z_norm = float(best["centroid_z"] / max(1, pet.shape[0] - 1))
    best["z_norm"] = z_norm
    best["side"] = qc_side_from_z_norm(z_norm, cfg.pet_side_fraction)
    return best, diagnostics


def qc_weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * float(cumulative[-1])
    return float(values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def find_ct_head_end_anchor(
    ct_arr: np.ndarray,
    body_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    cfg: HeadEndQcConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    values = ct_arr[body_mask & np.isfinite(ct_arr)]
    max_ct = float(values.max()) if values.size else float("nan")
    diagnostics: dict[str, Any] = {
        "ct_body_voxels": int(values.size),
        "ct_max_hu": max_ct,
        "ct_threshold_hu": float(cfg.ct_very_high_threshold_hu),
        "ct_components_raw": 0,
        "ct_components_kept": 0,
    }
    if values.size == 0 or max_ct < cfg.ct_very_high_threshold_hu:
        diagnostics["ct_reason"] = "ct_max_below_threshold"
        return None, diagnostics

    very_high = body_mask & np.isfinite(ct_arr) & (ct_arr >= cfg.ct_very_high_threshold_hu)
    records = qc_connected_component_records(very_high, spacing_xyz)
    diagnostics["ct_components_raw"] = len(records)
    kept: list[dict[str, Any]] = []
    for record in records:
        z_extent_mm = float((record["bbox_z1"] - record["bbox_z0"]) * spacing_xyz[2])
        record = dict(record)
        record["z_extent_mm"] = z_extent_mm
        if record["volume_mm3"] < cfg.ct_component_min_volume_mm3:
            continue
        if z_extent_mm > cfg.ct_component_max_z_extent_mm:
            continue
        kept.append(record)
    diagnostics["ct_components_kept"] = len(kept)
    if not kept:
        diagnostics["ct_reason"] = "no_component_after_filter"
        return None, diagnostics

    z_size = int(ct_arr.shape[0])
    z_extent_mm = float(z_size * spacing_xyz[2])
    band_mm = min(float(cfg.ct_band_extent_mm), float(cfg.ct_band_max_fraction) * z_extent_mm)
    band_voxels = min(max(1, int(round(band_mm / spacing_xyz[2]))), max(1, z_size // 2))
    volumes = np.array([r["volume_mm3"] for r in kept], dtype=np.float64)
    centroids = np.array([r["centroid_z"] for r in kept], dtype=np.float64)
    total_mass = float(volumes.sum())
    start_mass = float(volumes[centroids < band_voxels].sum())
    end_mass = float(volumes[centroids >= z_size - band_voxels].sum())
    median_z = qc_weighted_median(centroids, volumes)
    median_norm = float(median_z / max(1, z_size - 1))
    if start_mass >= end_mass:
        dominant_side = "start"
        dominant_mass = start_mass
        other_mass = end_mass
        median_ok = median_norm <= cfg.ct_median_side_fraction
    else:
        dominant_side = "end"
        dominant_mass = end_mass
        other_mass = start_mass
        median_ok = median_norm >= 1.0 - cfg.ct_median_side_fraction

    dominant_fraction = float(dominant_mass / max(total_mass, 1e-6))
    dominant_ratio = float(dominant_mass / max(other_mass, 1e-6))
    side = "unknown"
    reason = "not_concentrated"
    if dominant_fraction >= cfg.ct_dominant_fraction and dominant_ratio >= cfg.ct_dominant_ratio and median_ok:
        side = dominant_side
        reason = "concentrated"

    anchor = {
        "side": side,
        "reason": reason,
        "component_count": len(kept),
        "total_mass_mm3": total_mass,
        "start_mass_mm3": start_mass,
        "end_mass_mm3": end_mass,
        "dominant_side": dominant_side,
        "dominant_fraction": dominant_fraction,
        "dominant_ratio": dominant_ratio,
        "weighted_median_z": median_z,
        "weighted_median_z_norm": median_norm,
        "band_mm": float(band_mm),
        "band_voxels": int(band_voxels),
    }
    return anchor if side != "unknown" else None, {**diagnostics, **anchor}


def evaluate_no_label_head_end(ct: sitk.Image, pet: sitk.Image, cfg: HeadEndQcConfig) -> dict[str, Any]:
    if ct.GetSize() != pet.GetSize():
        raise RuntimeError("CT/PET size mismatch for head-end QC")
    spacing_xyz = tuple(float(x) for x in ct.GetSpacing())
    ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
    pet_arr = sitk.GetArrayFromImage(pet).astype(np.float32)
    body = make_head_end_qc_body_mask(ct_arr, cfg)
    pet_anchor, pet_diag = find_pet_head_end_anchor(pet_arr, body, spacing_xyz, cfg)
    ct_anchor, ct_diag = find_ct_head_end_anchor(ct_arr, body, spacing_xyz, cfg)
    pet_side = None if pet_anchor is None else str(pet_anchor.get("side", "unknown"))
    ct_side = None if ct_anchor is None else str(ct_anchor.get("side", "unknown"))
    if pet_side is not None and pet_side != "unknown":
        decision_source = "pet_brain_anchor"
        predicted_head_side = pet_side
    elif ct_side is not None and ct_side != "unknown":
        decision_source = "ct_very_high_component_anchor"
        predicted_head_side = ct_side
    else:
        decision_source = "fallback_unknown"
        predicted_head_side = "unknown"
    evidence_disagreement = (
        pet_side is not None
        and pet_side != "unknown"
        and ct_side is not None
        and ct_side != "unknown"
        and pet_side != ct_side
    )
    return {
        "decision_source": decision_source,
        "predicted_head_side": predicted_head_side,
        "evidence_conflict": bool(evidence_disagreement),
        "evidence_disagreement": bool(evidence_disagreement),
        "policy": "pet_first_ct_backup",
        "pet_anchor": pet_anchor,
        "pet_diag": pet_diag,
        "ct_anchor": ct_anchor,
        "ct_diag": ct_diag,
    }


def apply_wide_hnroi_z_cap(
    hn_ct: sitk.Image,
    hn_pet: sitk.Image,
    hn_bbox_xyz: tuple[int, int, int, int, int, int],
    cfg: WideHnRoiConfig,
) -> tuple[tuple[int, int, int, int, int, int], dict[str, Any]]:
    size = hn_ct.GetSize()
    spacing = hn_ct.GetSpacing()
    z_extent_mm = float(size[2] * spacing[2])
    info: dict[str, Any] = {
        "wide_hnroi_enabled": True,
        "wide_hnroi_source_extent_z_mm": z_extent_mm,
        "wide_hnroi_max_z_extent_mm": float(cfg.max_z_extent_mm),
        "wide_hnroi_action": "kept_unchanged",
    }
    if z_extent_mm <= cfg.max_z_extent_mm + 1e-6:
        return hn_bbox_xyz, info

    qc = evaluate_no_label_head_end(hn_ct, hn_pet, cfg.head_end_qc)
    predicted_side = str(qc["predicted_head_side"])
    info["head_end_qc"] = qc
    info["wide_hnroi_head_end_decision_source"] = qc["decision_source"]
    info["wide_hnroi_predicted_head_side"] = predicted_side
    if predicted_side != cfg.preserve_head_side:
        info["wide_hnroi_action"] = (
            "fallback_keep_full_hnroi"
            if predicted_side == "unknown"
            else "direction_conflict_keep_full_hnroi"
        )
        return hn_bbox_xyz, info

    keep_slices = max(1, min(int(size[2]), int(math.floor(cfg.max_z_extent_mm / spacing[2]))))
    if cfg.preserve_head_side == "end":
        local_z0 = int(size[2] - keep_slices)
        local_z1 = int(size[2])
    else:
        local_z0 = 0
        local_z1 = int(keep_slices)
    x0, x1, y0, y1, z0, _ = hn_bbox_xyz
    capped_bbox = (x0, x1, y0, y1, z0 + local_z0, z0 + local_z1)
    info["wide_hnroi_action"] = "z_cropped_keep_head_end"
    info["wide_hnroi_local_z0"] = int(local_z0)
    info["wide_hnroi_local_z1"] = int(local_z1)
    info["wide_hnroi_target_extent_z_mm"] = float((local_z1 - local_z0) * spacing[2])
    return capped_bbox, info


def expand_axis_to_min_size(low: int, high: int, min_size: int, axis_size: int) -> tuple[int, int]:
    current = high - low
    if current >= min_size:
        return low, high
    extra = min_size - current
    left = extra // 2
    right = extra - left
    low -= left
    high += right
    if low < 0:
        high = min(axis_size, high - low)
        low = 0
    if high > axis_size:
        low = max(0, low - (high - axis_size))
        high = axis_size
    return low, high


def resize_to_shape(arr: np.ndarray, target_shape: tuple[int, int, int], order: int) -> np.ndarray:
    zoom = tuple(target_shape[i] / arr.shape[i] for i in range(3))
    return ndi.zoom(arr, zoom=zoom, order=order)


def normalize_ct_for_tinyunet(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct.astype(np.float32), -200.0, 300.0)
    return (ct + 200.0) / 500.0 * 2.0 - 1.0


def normalize_pet_for_tinyunet(pet: np.ndarray) -> np.ndarray:
    pet = np.nan_to_num(pet.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pet = np.clip(pet, 0.0, 20.0)
    return pet / 10.0 - 1.0


def lowres_spacing_zyx(
    lowres_shape: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(spacing_xyz[2]) * float(original_shape[0]) / float(lowres_shape[0]),
        float(spacing_xyz[1]) * float(original_shape[1]) / float(lowres_shape[1]),
        float(spacing_xyz[0]) * float(original_shape[2]) / float(lowres_shape[2]),
    )


def bbox_gap_mm(
    bbox_a: tuple[int, int, int, int, int, int],
    bbox_b: tuple[int, int, int, int, int, int],
    spacing_zyx: tuple[float, float, float],
) -> float:
    az0, az1, ay0, ay1, ax0, ax1 = bbox_a
    bz0, bz1, by0, by1, bx0, bx1 = bbox_b
    gaps = (
        max(0, max(bz0 - az1, az0 - bz1)) * spacing_zyx[0],
        max(0, max(by0 - ay1, ay0 - by1)) * spacing_zyx[1],
        max(0, max(bx0 - ax1, ax0 - bx1)) * spacing_zyx[2],
    )
    return float(math.sqrt(sum(gap * gap for gap in gaps)))


def clean_lowres_components(
    mask: np.ndarray,
    prob: np.ndarray,
    original_shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    cfg: LocalRoiConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask, {
            "component_cleaning": True,
            "components_before": 0,
            "components_after": 0,
            "components_removed": 0,
            "voxels_removed": 0,
        }

    labeled, num_components = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if num_components <= 1:
        return mask, {
            "component_cleaning": True,
            "components_before": int(num_components),
            "components_after": int(num_components),
            "components_removed": 0,
            "voxels_removed": 0,
        }

    spacing_zyx = lowres_spacing_zyx(mask.shape, original_shape, spacing_xyz)
    components: list[dict[str, Any]] = []
    for label_index, slices in enumerate(ndi.find_objects(labeled), start=1):
        if slices is None:
            continue
        component_mask = labeled[slices] == label_index
        voxel_count = int(component_mask.sum())
        if voxel_count <= 0:
            continue
        prob_values = prob[slices][component_mask]
        z_slice, y_slice, x_slice = slices
        components.append(
            {
                "label": label_index,
                "voxels": voxel_count,
                "bbox": (
                    int(z_slice.start),
                    int(z_slice.stop),
                    int(y_slice.start),
                    int(y_slice.stop),
                    int(x_slice.start),
                    int(x_slice.stop),
                ),
                "max_prob": float(prob_values.max()),
                "mean_prob": float(prob_values.mean()),
            }
        )

    if not components:
        return np.zeros_like(mask, dtype=bool), {
            "component_cleaning": True,
            "components_before": int(num_components),
            "components_after": 0,
            "components_removed": int(num_components),
            "voxels_removed": int(mask.sum()),
        }

    components.sort(key=lambda item: (item["voxels"], item["max_prob"]), reverse=True)
    largest_voxels = int(components[0]["voxels"])
    keep_labels: set[int] = set()
    anchor_components: list[dict[str, Any]] = []

    for rank, component in enumerate(components):
        large_enough = (
            component["voxels"] >= cfg.component_min_voxels
            and component["voxels"] >= cfg.component_min_relative_size * largest_voxels
        )
        top_ranked = rank < cfg.component_keep_top_k
        high_confidence = component["max_prob"] >= cfg.component_keep_max_prob
        if rank == 0 or large_enough or top_ranked or high_confidence:
            keep_labels.add(int(component["label"]))
            anchor_components.append(component)

    for component in components:
        label = int(component["label"])
        if label in keep_labels:
            continue
        nearest_anchor_gap = min(
            bbox_gap_mm(component["bbox"], anchor["bbox"], spacing_zyx) for anchor in anchor_components
        )
        if nearest_anchor_gap <= cfg.component_far_gap_mm:
            keep_labels.add(label)

    cleaned = np.isin(labeled, list(keep_labels))
    components_after = len(keep_labels)
    return cleaned, {
        "component_cleaning": True,
        "components_before": int(len(components)),
        "components_after": int(components_after),
        "components_removed": int(len(components) - components_after),
        "voxels_removed": int(mask.sum() - cleaned.sum()),
    }


def bbox_from_prob_threshold(
    prob: np.ndarray,
    threshold: float,
    original_shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    cfg: LocalRoiConfig,
) -> tuple[tuple[int, int, int, int, int, int] | None, dict[str, Any]]:
    mask_lr = prob >= threshold
    mask_lr, clean_stats = clean_lowres_components(mask_lr, prob, original_shape, spacing_xyz, cfg)
    return bbox_from_mask_zyx(mask_lr), clean_stats


def lowres_bbox_to_original(
    bbox_lr: tuple[int, int, int, int, int, int],
    lr_shape: tuple[int, int, int],
    original_shape: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox_lr
    scale_z = original_shape[0] / lr_shape[0]
    scale_y = original_shape[1] / lr_shape[1]
    scale_x = original_shape[2] / lr_shape[2]
    bbox = (
        int(math.floor(z0 * scale_z)),
        int(math.ceil(z1 * scale_z)),
        int(math.floor(y0 * scale_y)),
        int(math.ceil(y1 * scale_y)),
        int(math.floor(x0 * scale_x)),
        int(math.ceil(x1 * scale_x)),
    )
    return clamp_bbox_zyx(bbox, original_shape)


def add_tinyunet_margin_and_min_size(
    bbox: tuple[int, int, int, int, int, int],
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    cfg: LocalRoiConfig,
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    mx = int(math.ceil(cfg.xy_margin_mm / spacing_xyz[0]))
    my = int(math.ceil(cfg.xy_margin_mm / spacing_xyz[1]))
    mz = int(math.ceil(cfg.z_margin_mm / spacing_xyz[2]))
    z0, z1, y0, y1, x0, x1 = clamp_bbox_zyx(
        (z0 - mz, z1 + mz, y0 - my, y1 + my, x0 - mx, x1 + mx),
        shape_zyx,
    )
    min_x = int(math.ceil(cfg.min_xy_size_mm / spacing_xyz[0]))
    min_y = int(math.ceil(cfg.min_xy_size_mm / spacing_xyz[1]))
    min_z = int(math.ceil(cfg.min_z_size_mm / spacing_xyz[2]))
    z0, z1 = expand_axis_to_min_size(z0, z1, min_z, shape_zyx[0])
    y0, y1 = expand_axis_to_min_size(y0, y1, min_y, shape_zyx[1])
    x0, x1 = expand_axis_to_min_size(x0, x1, min_x, shape_zyx[2])
    if cfg.max_z_extent_mm and cfg.max_z_extent_mm > 0:
        max_z = max(min_z, int(math.floor(cfg.max_z_extent_mm / spacing_xyz[2])))
        if z1 - z0 > max_z:
            z1 = min(shape_zyx[0], z0 + max_z)
    return clamp_bbox_zyx((z0, z1, y0, y1, x0, x1), shape_zyx)


def volume_from_bbox_zyx(bbox: tuple[int, int, int, int, int, int]) -> int:
    z0, z1, y0, y1, x0, x1 = bbox
    return int((z1 - z0) * (y1 - y0) * (x1 - x0))


def bbox_zyx_to_xyz(bbox: tuple[int, int, int, int, int, int]) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    return x0, x1, y0, y1, z0, z1


def local_roi_candidate_from_probability(
    prob_lr: np.ndarray,
    reference: sitk.Image,
    threshold: float,
    cfg: LocalRoiConfig,
) -> dict[str, Any]:
    shape_zyx = tuple(reversed(reference.GetSize()))
    spacing_xyz = reference.GetSpacing()
    bbox_lr, clean_stats = bbox_from_prob_threshold(prob_lr, threshold, shape_zyx, spacing_xyz, cfg)
    hn_voxels = int(np.prod(shape_zyx))
    if bbox_lr is None:
        bbox_zyx = (0, shape_zyx[0], 0, shape_zyx[1], 0, shape_zyx[2])
        method = "full_hnroi_tinyunet_empty"
    else:
        bbox_zyx = lowres_bbox_to_original(bbox_lr, prob_lr.shape, shape_zyx)
        bbox_zyx = add_tinyunet_margin_and_min_size(bbox_zyx, shape_zyx, spacing_xyz, cfg)
        method = "tinyunet_lowres_adaptive_cleaned"

    roi_voxels = volume_from_bbox_zyx(bbox_zyx)
    z_mm = (bbox_zyx[1] - bbox_zyx[0]) * spacing_xyz[2]
    fallback_recommended = bool(
        bbox_lr is None
        or roi_voxels / hn_voxels < cfg.fallback_min_roi_fraction
        or z_mm < cfg.fallback_min_z_mm
    )
    row: dict[str, Any] = {
        "local_roi_method": method,
        "threshold": float(threshold),
        "bbox_zyx": bbox_zyx,
        "local_bbox_xyz_hnroi": bbox_zyx_to_xyz(bbox_zyx),
        "roi_voxels": int(roi_voxels),
        "hn_voxels": int(hn_voxels),
        "roi_to_hn_ratio": float(roi_voxels / hn_voxels),
        "fallback_recommended": fallback_recommended,
        "prob_max": float(np.nanmax(prob_lr)) if prob_lr.size else 0.0,
        "prob_p999": float(np.nanpercentile(prob_lr, 99.9)) if prob_lr.size else 0.0,
        "coarse_foreground_voxels": 0 if bbox_lr is None else int((prob_lr >= threshold).sum()),
        "raw_bbox_lr_zyx": None if bbox_lr is None else bbox_lr,
        **clean_stats,
    }
    return row


def no_label_qc_failure_reasons(row: dict[str, Any], cfg: LocalRoiConfig, spacing_z: float) -> list[str]:
    reasons: list[str] = []
    if row.get("local_roi_method") == "full_hnroi_tinyunet_empty":
        reasons.append("empty_mask")
    if bool(row.get("fallback_recommended", False)):
        reasons.append("fallback_recommended")

    roi_ratio = float(row.get("roi_to_hn_ratio", 1.0))
    if roi_ratio < cfg.qc_min_roi_fraction:
        reasons.append("roi_too_small")
    if roi_ratio > cfg.qc_max_roi_fraction:
        reasons.append("roi_too_large")

    prob_max = float(row.get("prob_max", 0.0))
    if prob_max < cfg.qc_min_prob_max:
        reasons.append("prob_max_too_low")
    if prob_max < cfg.qc_low_conf_prob_max and roi_ratio < cfg.qc_low_conf_min_roi_fraction:
        reasons.append("low_conf_roi_too_small")

    components_after = int(row.get("components_after") or 0)
    if components_after <= 0:
        reasons.append("no_components_after_cleaning")
    if components_after > cfg.qc_max_components_after:
        reasons.append("too_many_components_after_cleaning")

    z0, z1, _, _, _, _ = row["bbox_zyx"]
    z_mm = (z1 - z0) * spacing_z
    if z_mm < cfg.qc_min_z_mm:
        reasons.append("z_extent_too_small")
    if z_mm > cfg.qc_max_z_mm:
        reasons.append("z_extent_too_large")
    return reasons


def is_nonfallback_local_roi(row: dict[str, Any]) -> bool:
    if row.get("local_roi_method") == "full_hnroi_tinyunet_empty":
        return False
    if bool(row.get("fallback_recommended", False)):
        return False
    return float(row.get("roi_to_hn_ratio", 1.0)) < 0.999


def select_adaptive_local_roi(
    rows: list[dict[str, Any]],
    reference: sitk.Image,
    cfg: LocalRoiConfig,
) -> dict[str, Any]:
    spacing_z = float(reference.GetSpacing()[2])
    for row in rows:
        reasons = no_label_qc_failure_reasons(row, cfg, spacing_z)
        if not reasons:
            row = dict(row)
            row["adaptive_selection_reason"] = "first_no_label_qc_pass"
            row["no_label_qc_failure_reasons"] = ""
            return row
    for row in rows:
        if is_nonfallback_local_roi(row):
            selected = dict(row)
            selected["adaptive_selection_reason"] = "first_nonfallback_qc_relaxed"
            selected["no_label_qc_failure_reasons"] = ",".join(no_label_qc_failure_reasons(row, cfg, spacing_z))
            return selected
    selected = dict(rows[0])
    selected["adaptive_selection_reason"] = "fallback_full_hn_no_nonempty_candidate"
    selected["no_label_qc_failure_reasons"] = ",".join(no_label_qc_failure_reasons(rows[0], cfg, spacing_z))
    return selected


def local_roi_from_tinyunet_probability(
    prob_lr: np.ndarray,
    reference: sitk.Image,
    cfg: LocalRoiConfig,
) -> tuple[tuple[int, int, int, int, int, int], dict[str, Any]]:
    rows = [local_roi_candidate_from_probability(prob_lr, reference, threshold, cfg) for threshold in cfg.threshold_candidates]
    selected = select_adaptive_local_roi(rows, reference, cfg)
    selected["threshold_candidates"] = list(cfg.threshold_candidates)
    selected["tinyunet_probability_shape_zyx"] = tuple(int(v) for v in prob_lr.shape)
    return selected["local_bbox_xyz_hnroi"], selected


def write_nnunet_inputs(ct: sitk.Image, pet: sitk.Image, out_dir: Path, case_id: str = CASE_ID) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ct = sitk.Cast(ct, sitk.sitkFloat32)
    pet = sitk.Cast(pet, sitk.sitkFloat32)
    ct_path = out_dir / f"{case_id}_0000.nii.gz"
    pet_path = out_dir / f"{case_id}_0001.nii.gz"
    sitk.WriteImage(ct, str(ct_path), useCompression=True)
    sitk.WriteImage(pet, str(pet_path), useCompression=True)
    return [str(ct_path), str(pet_path)]


def image_to_nnunet_array(ct: sitk.Image, pet: sitk.Image) -> tuple[np.ndarray, dict[str, Any]]:
    pet = resample_to_reference(pet, ct, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    ct_arr = sitk.GetArrayFromImage(sitk.Cast(ct, sitk.sitkFloat32)).astype(np.float32, copy=False)
    pet_arr = sitk.GetArrayFromImage(sitk.Cast(pet, sitk.sitkFloat32)).astype(np.float32, copy=False)
    data = np.stack([ct_arr, pet_arr], axis=0)
    spacing_zyx = np.asarray([ct.GetSpacing()[2], ct.GetSpacing()[1], ct.GetSpacing()[0]], dtype=np.float64)
    return data, {"spacing": spacing_zyx}


def image_to_nnunet_array_with_priors(
    ct: sitk.Image,
    pet: sitk.Image,
    prior_seg: sitk.Image,
) -> tuple[np.ndarray, dict[str, Any]]:
    pet = resample_to_reference(pet, ct, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    prior_seg = resample_to_reference(prior_seg, ct, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    ct_arr = sitk.GetArrayFromImage(sitk.Cast(ct, sitk.sitkFloat32)).astype(np.float32, copy=False)
    pet_arr = sitk.GetArrayFromImage(sitk.Cast(pet, sitk.sitkFloat32)).astype(np.float32, copy=False)
    prior_arr = sitk.GetArrayFromImage(sitk.Cast(prior_seg, sitk.sitkUInt8)).astype(np.uint8, copy=False)
    gtvp_prior = (prior_arr == 1).astype(np.float32, copy=False)
    gtvn_prior = (prior_arr == 2).astype(np.float32, copy=False)
    data = np.stack([ct_arr, pet_arr, gtvp_prior, gtvn_prior], axis=0)
    spacing_zyx = np.asarray([ct.GetSpacing()[2], ct.GetSpacing()[1], ct.GetSpacing()[0]], dtype=np.float64)
    return data, {"spacing": spacing_zyx}


def bbox_from_segmentation_with_margin(
    seg: sitk.Image,
    reference: sitk.Image,
    cfg: RefinementRoiConfig,
) -> tuple[tuple[int, int, int, int, int, int], dict[str, Any]]:
    seg = resample_to_reference(seg, reference, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    seg_arr = sitk.GetArrayFromImage(sitk.Cast(seg, sitk.sitkUInt8)).astype(np.uint8, copy=False)
    shape_zyx = tuple(int(v) for v in reversed(reference.GetSize()))
    spacing_xyz = reference.GetSpacing()
    raw_bbox = bbox_from_mask_zyx(seg_arr > 0)
    if raw_bbox is None:
        bbox_zyx = (0, shape_zyx[0], 0, shape_zyx[1], 0, shape_zyx[2])
        info = {
            "local_roi_method": "dataset807_refine_full_hnroi_empty_dataset804_prior",
            "prior_foreground_voxels": 0,
            "prior_gtvp_voxels": int((seg_arr == 1).sum()),
            "prior_gtvn_voxels": int((seg_arr == 2).sum()),
        }
        return bbox_zyx_to_xyz(bbox_zyx), info

    z0, z1, y0, y1, x0, x1 = raw_bbox
    mx = int(math.ceil(cfg.margin_xyz_mm[0] / spacing_xyz[0]))
    my = int(math.ceil(cfg.margin_xyz_mm[1] / spacing_xyz[1]))
    mz = int(math.ceil(cfg.margin_xyz_mm[2] / spacing_xyz[2]))
    bbox_zyx = clamp_bbox_zyx((z0 - mz, z1 + mz, y0 - my, y1 + my, x0 - mx, x1 + mx), shape_zyx)
    z0, z1, y0, y1, x0, x1 = bbox_zyx
    min_x = int(math.ceil(cfg.min_extent_xyz_mm[0] / spacing_xyz[0]))
    min_y = int(math.ceil(cfg.min_extent_xyz_mm[1] / spacing_xyz[1]))
    min_z = int(math.ceil(cfg.min_extent_xyz_mm[2] / spacing_xyz[2]))
    z0, z1 = expand_axis_to_min_size(z0, z1, min_z, shape_zyx[0])
    y0, y1 = expand_axis_to_min_size(y0, y1, min_y, shape_zyx[1])
    x0, x1 = expand_axis_to_min_size(x0, x1, min_x, shape_zyx[2])
    bbox_zyx = clamp_bbox_zyx((z0, z1, y0, y1, x0, x1), shape_zyx)
    local_bbox_xyz = bbox_zyx_to_xyz(bbox_zyx)
    info = {
        "local_roi_method": "dataset807_refine_bbox_from_dataset804_cascade_tta_prior",
        "raw_prior_bbox_zyx": raw_bbox,
        "bbox_zyx": bbox_zyx,
        "local_bbox_xyz_hnroi": local_bbox_xyz,
        "margin_xyz_mm": list(cfg.margin_xyz_mm),
        "min_extent_xyz_mm": list(cfg.min_extent_xyz_mm),
        "prior_foreground_voxels": int((seg_arr > 0).sum()),
        "prior_gtvp_voxels": int((seg_arr == 1).sum()),
        "prior_gtvn_voxels": int((seg_arr == 2).sum()),
    }
    return local_bbox_xyz, info


def array_to_label_image(seg_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    seg = np.rint(np.asarray(seg_zyx)).astype(np.uint8)
    image = sitk.GetImageFromArray(seg)
    image.CopyInformation(reference)
    return image


EMPTY_TINY_MIN_TOTAL_VOLUME_ML = 0.05
EMPTY_TINY_MIN_COMPONENT_VOLUME_ML = 0.05
EMPTY_TINY_MAX_FALLBACK_VOLUME_ML = 80.0
EMPTY_TINY_TARGET_FALLBACK_VOLUME_ML = 5.0
EMPTY_TINY_FALLBACK_THRESHOLDS = (0.5, 0.2, 0.02, 0.002, 0.001, 0.0005, 0.0001, 0.00005)


def crop_array_by_bbox_xyz(array_zyx: np.ndarray, bbox_xyz: tuple[int, int, int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bbox_xyz
    return np.asarray(array_zyx)[z0:z1, y0:y1, x0:x1]


def crop_probabilities_by_bbox_xyz(
    probabilities_czyx: np.ndarray | None,
    bbox_xyz: tuple[int, int, int, int, int, int],
) -> np.ndarray | None:
    if probabilities_czyx is None:
        return None
    probs = np.asarray(probabilities_czyx)
    if probs.ndim != 4:
        return None
    x0, x1, y0, y1, z0, z1 = bbox_xyz
    return probs[:, z0:z1, y0:y1, x0:x1]


def mask_qc_reason(mask_zyx: np.ndarray, reference: sitk.Image) -> dict[str, Any]:
    foreground = np.asarray(mask_zyx > 0, dtype=bool)
    spacing = reference.GetSpacing()
    voxel_ml = float(spacing[0] * spacing[1] * spacing[2] / 1000.0)
    total_ml = float(foreground.sum() * voxel_ml)
    if not foreground.any():
        return {
            "failed": True,
            "reason": "empty_mask",
            "total_volume_ml": 0.0,
            "largest_component_volume_ml": 0.0,
            "component_count": 0,
            "valid_component_count": 0,
        }
    components, n_components = ndi.label(foreground, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(components.ravel())[1:] if n_components else np.asarray([], dtype=np.int64)
    volumes_ml = sizes.astype(np.float64) * voxel_ml
    largest_ml = float(volumes_ml.max()) if volumes_ml.size else 0.0
    valid_count = int(np.count_nonzero(volumes_ml >= EMPTY_TINY_MIN_COMPONENT_VOLUME_ML))
    reasons: list[str] = []
    if total_ml < EMPTY_TINY_MIN_TOTAL_VOLUME_ML:
        reasons.append("total_volume_tiny")
    if valid_count < 1:
        reasons.append("no_valid_component")
    return {
        "failed": bool(reasons),
        "reason": ";".join(reasons) if reasons else "ok",
        "total_volume_ml": total_ml,
        "largest_component_volume_ml": largest_ml,
        "component_count": int(n_components),
        "valid_component_count": valid_count,
    }


def probability_to_labeled_support(probabilities_czyx: np.ndarray, support_zyx: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities_czyx, dtype=np.float32)
    support = np.asarray(support_zyx, dtype=bool)
    out = np.zeros(support.shape, dtype=np.uint8)
    if probs.shape[0] >= 3:
        class_1 = probs[1]
        class_2 = probs[2]
        out[support] = np.where(class_2[support] > class_1[support], 2, 1).astype(np.uint8)
    else:
        out[support] = 1
    return out


def fallback_candidate_from_probability(
    probabilities_czyx: np.ndarray | None,
    reference: sitk.Image,
    source: str,
    allow_weak_last_resort: bool = True,
) -> dict[str, Any] | None:
    if probabilities_czyx is None:
        return None
    probs = np.asarray(probabilities_czyx, dtype=np.float32)
    if probs.ndim != 4 or probs.shape[0] < 2:
        return None
    ref_shape = tuple(reversed(reference.GetSize()))
    if tuple(probs.shape[1:]) != ref_shape:
        debug_log(f"skip {source} fallback: probability shape {probs.shape[1:]} != reference shape {ref_shape}")
        return None

    spacing = reference.GetSpacing()
    voxel_ml = float(spacing[0] * spacing[1] * spacing[2] / 1000.0)
    foreground_prob = np.clip(np.nan_to_num(probs[1:].sum(axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    accepted: list[dict[str, Any]] = []
    weak: dict[str, Any] | None = None
    for threshold in EMPTY_TINY_FALLBACK_THRESHOLDS:
        support = foreground_prob >= float(threshold)
        if not support.any():
            continue
        comp, n_components = ndi.label(support, structure=np.ones((3, 3, 3), dtype=bool))
        if n_components < 1:
            continue
        slices = ndi.find_objects(comp)
        components: list[dict[str, float]] = []
        for label_idx, bbox in enumerate(slices, start=1):
            if bbox is None:
                continue
            component = comp[bbox] == label_idx
            volume_ml = float(component.sum() * voxel_ml)
            if volume_ml < EMPTY_TINY_MIN_COMPONENT_VOLUME_ML:
                continue
            values = foreground_prob[bbox][component]
            components.append(
                {
                    "label": float(label_idx),
                    "volume_ml": volume_ml,
                    "max_prob": float(values.max()) if values.size else 0.0,
                }
            )
        if not components:
            continue
        components.sort(key=lambda item: (item["max_prob"], item["volume_ml"]), reverse=True)
        keep_labels = [int(item["label"]) for item in components[:6]]
        kept = np.isin(comp, keep_labels)
        candidate = probability_to_labeled_support(probs, kept)
        qc = mask_qc_reason(candidate, reference)
        candidate_volume_ml = float(kept.sum() * voxel_ml)
        row = {
            "array": candidate,
            "source": source,
            "threshold": float(threshold),
            "total_volume_ml": candidate_volume_ml,
            "largest_component_volume_ml": float(qc["largest_component_volume_ml"]),
            "component_count": int(qc["component_count"]),
            "accepted": bool(
                candidate_volume_ml >= EMPTY_TINY_MIN_TOTAL_VOLUME_ML
                and candidate_volume_ml <= EMPTY_TINY_MAX_FALLBACK_VOLUME_ML
                and int(qc["valid_component_count"]) >= 1
            ),
        }
        if row["accepted"]:
            accepted.append(row)
        elif allow_weak_last_resort and candidate_volume_ml > 0.0 and weak is None:
            weak = row

    if accepted:
        target = max(EMPTY_TINY_TARGET_FALLBACK_VOLUME_ML, EMPTY_TINY_MIN_TOTAL_VOLUME_ML)

        def score(item: dict[str, Any]) -> tuple[float, float]:
            volume = max(float(item["total_volume_ml"]), 1e-6)
            return (-abs(math.log(volume / target)), float(item["threshold"]))

        accepted.sort(key=score, reverse=True)
        return accepted[0]
    return weak


def fallback_candidate_from_mask(mask_zyx: np.ndarray | None, reference: sitk.Image, source: str) -> dict[str, Any] | None:
    if mask_zyx is None:
        return None
    arr = np.rint(np.asarray(mask_zyx)).astype(np.uint8)
    if tuple(arr.shape) != tuple(reversed(reference.GetSize())):
        debug_log(f"skip {source} fallback: mask shape {arr.shape} != reference shape {tuple(reversed(reference.GetSize()))}")
        return None
    qc = mask_qc_reason(arr, reference)
    if qc["failed"]:
        return None
    return {
        "array": arr,
        "source": source,
        "threshold": None,
        "total_volume_ml": float(qc["total_volume_ml"]),
        "largest_component_volume_ml": float(qc["largest_component_volume_ml"]),
        "component_count": int(qc["component_count"]),
        "accepted": True,
    }


def load_tinyunet_ensemble(model_dir: Path, cfg: LocalRoiConfig, device: torch.device) -> list[nn.Module]:
    models: list[nn.Module] = []
    for fold in range(5):
        checkpoint = model_dir / f"fold_{fold}" / "checkpoint_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"TinyUNet checkpoint missing: {checkpoint}")
        model = make_tinyunet_model(cfg.base_channels, cfg.downsample_levels).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        model.eval()
        models.append(model)
    return models


def make_tinyunet_input_array(
    ct: sitk.Image,
    pet: sitk.Image,
    cfg: LocalRoiConfig,
) -> np.ndarray:
    pet = resample_to_reference(pet, ct, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    ct_arr = sitk.GetArrayFromImage(sitk.Cast(ct, sitk.sitkFloat32)).astype(np.float32, copy=False)
    pet_arr = sitk.GetArrayFromImage(sitk.Cast(pet, sitk.sitkFloat32)).astype(np.float32, copy=False)
    ct_lr = resize_to_shape(normalize_ct_for_tinyunet(ct_arr), cfg.target_shape_zyx, order=1).astype(np.float32)
    pet_lr = resize_to_shape(normalize_pet_for_tinyunet(pet_arr), cfg.target_shape_zyx, order=1).astype(np.float32)
    return np.stack([ct_lr, pet_lr], axis=0).astype(np.float32, copy=False)


def tinyunet_input_tensor(
    image_ncdhw: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(image_ncdhw[None]).to(device, non_blocking=True)


@torch.no_grad()
def predict_tinyunet_ensemble_prob(
    models: list[nn.Module],
    image_ncdhw: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = tinyunet_input_tensor(image_ncdhw, device)
    probs: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    for model in models:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0]
        probs.append(prob.detach().float().cpu().numpy())
    return np.mean(probs, axis=0).astype(np.float32)


@torch.no_grad()
def predict_tinyunet_ensemble_prob_batch(
    models: list[nn.Module],
    images_ncdhw: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    if not images_ncdhw:
        return []
    if len(images_ncdhw) == 1:
        return [predict_tinyunet_ensemble_prob(models, images_ncdhw[0], device)]
    batch = torch.from_numpy(np.stack(images_ncdhw, axis=0)).to(device, non_blocking=True)
    accum: torch.Tensor | None = None
    use_amp = device.type == "cuda"
    for model in models:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch)
            prob = torch.sigmoid(logits[:, 0]).float()
        accum = prob if accum is None else accum + prob
    mean_prob = (accum / float(len(models))).detach().cpu().numpy().astype(np.float32)
    return [mean_prob[i] for i in range(mean_prob.shape[0])]


def make_predictor(model_dir: Path, checkpoint_name: str) -> nnUNetPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tile_step_size = env_float("HECKTOR_FINE_TILE_STEP_SIZE", 0.8, minimum=0.05, maximum=1.0)
    fine_tile_batch_size = env_int("HECKTOR_FINE_TILE_BATCH_SIZE", 1, minimum=1)
    fine_use_tta = env_bool("HECKTOR_FINE_USE_TTA", True)
    fine_folds = env_folds("HECKTOR_FINE_FOLDS", (0, 1, 2, 3, 4))
    debug_log(f"fine nnUNet tile_step_size={fine_tile_step_size}")
    debug_log(f"fine nnUNet tile_batch_size={fine_tile_batch_size}")
    debug_log(f"fine nnUNet use_tta={fine_use_tta}")
    debug_log(f"fine nnUNet folds={fine_folds}")
    predictor_cls = BatchedTileNNUNetPredictor if fine_tile_batch_size > 1 else nnUNetPredictor
    predictor_kwargs = {"tile_batch_size": fine_tile_batch_size} if fine_tile_batch_size > 1 else {}
    predictor = predictor_cls(
        tile_step_size=fine_tile_step_size,
        use_gaussian=True,
        use_mirroring=fine_use_tta,
        perform_everything_on_device=torch.cuda.is_available(),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
        **predictor_kwargs,
    )
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=fine_folds, checkpoint_name=checkpoint_name)
    return predictor


def run_nnunet_prediction_in_memory(
    predictor: nnUNetPredictor,
    ct: sitk.Image,
    pet: sitk.Image,
    return_probabilities: bool,
) -> tuple[sitk.Image, np.ndarray | None]:
    data, properties = image_to_nnunet_array(ct, pet)
    return run_nnunet_prediction_from_array(predictor, data, properties, ct, return_probabilities)


def run_nnunet_prediction_from_array(
    predictor: nnUNetPredictor,
    data: np.ndarray,
    properties: dict[str, Any],
    reference: sitk.Image,
    return_probabilities: bool,
    segmentation_previous_stage: np.ndarray | None = None,
) -> tuple[sitk.Image, np.ndarray | None]:
    debug_log(f"nnUNet in-memory input shape={data.shape}; spacing_zyx={properties['spacing'].tolist()}")
    if return_probabilities:
        seg, prob = predictor.predict_single_npy_array(
            data,
            properties,
            segmentation_previous_stage=segmentation_previous_stage,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        return array_to_label_image(seg, reference), np.asarray(prob)
    seg = predictor.predict_single_npy_array(
        data,
        properties,
        segmentation_previous_stage=segmentation_previous_stage,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    return array_to_label_image(seg, reference), None


def run_nnunet_prediction(
    predictor: nnUNetPredictor,
    ct: sitk.Image,
    pet: sitk.Image,
    work_dir: Path,
    save_probabilities: bool,
) -> tuple[sitk.Image, Path]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    in_dir = work_dir / "in"
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = write_nnunet_inputs(ct, pet, in_dir)
    out_trunc = out_dir / CASE_ID
    predictor.predict_from_files(
        [inputs],
        [str(out_trunc)],
        save_probabilities=save_probabilities,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    seg_path = out_dir / f"{CASE_ID}.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"nnU-Net did not write {seg_path}")
    return sitk.ReadImage(str(seg_path)), out_dir


def finite_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def empty_region_features(prefix: str) -> dict[str, float]:
    features: dict[str, float] = {
        f"{prefix}_voxel_count": 0.0,
        f"{prefix}_volume_cc": 0.0,
        f"{prefix}_extent_z_mm": 0.0,
        f"{prefix}_extent_y_mm": 0.0,
        f"{prefix}_extent_x_mm": 0.0,
        f"{prefix}_max_extent_mm": 0.0,
        f"{prefix}_centroid_z_mm_from_center": 0.0,
        f"{prefix}_centroid_y_mm_from_center": 0.0,
        f"{prefix}_centroid_x_mm_from_center": 0.0,
        f"{prefix}_component_count": 0.0,
        f"{prefix}_largest_component_volume_cc": 0.0,
        f"{prefix}_largest_component_fraction": 0.0,
        f"{prefix}_largest_component_extent_z_mm": 0.0,
        f"{prefix}_largest_component_extent_y_mm": 0.0,
        f"{prefix}_largest_component_extent_x_mm": 0.0,
        f"{prefix}_pet_tlg": 0.0,
    }
    features.update(finite_stats(np.array([], dtype=np.float32), f"{prefix}_ct"))
    features.update(finite_stats(np.array([], dtype=np.float32), f"{prefix}_pet"))
    return features


def region_features(
    mask_bool: np.ndarray,
    ct: np.ndarray,
    pet: np.ndarray,
    spacing_zyx: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    voxel_count = int(mask_bool.sum())
    if voxel_count == 0:
        return empty_region_features(prefix)

    voxel_volume_cc = float(np.prod(spacing_zyx) / 1000.0)
    coords = np.argwhere(mask_bool)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents_mm = (maxs - mins + 1) * spacing_zyx
    centroid_vox = coords.mean(axis=0)
    image_center_vox = (np.asarray(mask_bool.shape, dtype=np.float64) - 1.0) / 2.0
    centroid_from_center_mm = (centroid_vox - image_center_vox) * spacing_zyx

    features: dict[str, float] = {
        f"{prefix}_voxel_count": float(voxel_count),
        f"{prefix}_volume_cc": float(voxel_count * voxel_volume_cc),
        f"{prefix}_extent_z_mm": float(extents_mm[0]),
        f"{prefix}_extent_y_mm": float(extents_mm[1]),
        f"{prefix}_extent_x_mm": float(extents_mm[2]),
        f"{prefix}_max_extent_mm": float(extents_mm.max()),
        f"{prefix}_centroid_z_mm_from_center": float(centroid_from_center_mm[0]),
        f"{prefix}_centroid_y_mm_from_center": float(centroid_from_center_mm[1]),
        f"{prefix}_centroid_x_mm_from_center": float(centroid_from_center_mm[2]),
    }
    features.update(finite_stats(ct[mask_bool], f"{prefix}_ct"))
    pet_values = pet[mask_bool]
    features.update(finite_stats(pet_values, f"{prefix}_pet"))
    finite_pet = pet_values[np.isfinite(pet_values)]
    features[f"{prefix}_pet_tlg"] = float(np.mean(finite_pet) * features[f"{prefix}_volume_cc"]) if finite_pet.size else 0.0

    comp, n_comp = ndi.label(mask_bool, structure=np.ones((3, 3, 3), dtype=bool))
    features[f"{prefix}_component_count"] = float(n_comp)
    if n_comp == 0:
        features[f"{prefix}_largest_component_volume_cc"] = 0.0
        features[f"{prefix}_largest_component_fraction"] = 0.0
        features[f"{prefix}_largest_component_extent_z_mm"] = 0.0
        features[f"{prefix}_largest_component_extent_y_mm"] = 0.0
        features[f"{prefix}_largest_component_extent_x_mm"] = 0.0
        return features

    comp_sizes = np.bincount(comp.ravel())[1:]
    largest_label = int(np.argmax(comp_sizes) + 1)
    largest_mask = comp == largest_label
    largest_coords = np.argwhere(largest_mask)
    largest_mins = largest_coords.min(axis=0)
    largest_maxs = largest_coords.max(axis=0)
    largest_extents_mm = (largest_maxs - largest_mins + 1) * spacing_zyx
    features[f"{prefix}_largest_component_volume_cc"] = float(int(largest_mask.sum()) * voxel_volume_cc)
    features[f"{prefix}_largest_component_fraction"] = float(largest_mask.sum() / voxel_count)
    features[f"{prefix}_largest_component_extent_z_mm"] = float(largest_extents_mm[0])
    features[f"{prefix}_largest_component_extent_y_mm"] = float(largest_extents_mm[1])
    features[f"{prefix}_largest_component_extent_x_mm"] = float(largest_extents_mm[2])
    return features


def extract_shallow_features(ct: sitk.Image, pet: sitk.Image, mask: sitk.Image) -> dict[str, float]:
    pet = resample_to_reference(pet, ct, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    mask = resample_to_reference(mask, ct, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
    pet_arr = sitk.GetArrayFromImage(pet).astype(np.float32)
    mask_arr = np.rint(sitk.GetArrayFromImage(mask)).astype(np.int16)
    spacing_zyx = np.asarray([ct.GetSpacing()[2], ct.GetSpacing()[1], ct.GetSpacing()[0]], dtype=np.float64)

    features: dict[str, float] = {}
    features.update(region_features(mask_arr == 1, ct_arr, pet_arr, spacing_zyx, "gtvp"))
    features.update(region_features(mask_arr == 2, ct_arr, pet_arr, spacing_zyx, "gtvn"))
    features.update(region_features(mask_arr > 0, ct_arr, pet_arr, spacing_zyx, "tumor_union"))
    features["gtvn_to_gtvp_volume_ratio"] = (
        features["gtvn_volume_cc"] / features["gtvp_volume_cc"] if features["gtvp_volume_cc"] > 0 else 0.0
    )
    features["tumor_total_component_count"] = features["gtvp_component_count"] + features["gtvn_component_count"]
    return features


def soft_threshold_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def soft_safe_percentiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    finite = finite[finite > SOFT_MIN_POSITIVE]
    for percentile in (50, 75, 90, 95, 99, 99.5, 99.9):
        key = str(percentile).replace(".", "p")
        out[f"{prefix}_prob_pos_p{key}"] = float(np.percentile(finite, percentile)) if finite.size else 0.0
    out[f"{prefix}_prob_pos_count"] = float(finite.size)
    return out


def soft_component_features(prob: np.ndarray, threshold: float, voxel_volume_cc: float, prefix: str) -> dict[str, float]:
    binary = prob >= threshold
    comp, n_comp = ndi.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    name = soft_threshold_name(threshold)
    out: dict[str, float] = {
        f"{prefix}_thr{name}_component_count": float(n_comp),
        f"{prefix}_thr{name}_binary_volume_cc": float(binary.sum() * voxel_volume_cc),
        f"{prefix}_thr{name}_soft_mass_cc": float(prob[binary].sum() * voxel_volume_cc),
        f"{prefix}_thr{name}_largest_component_volume_cc": 0.0,
        f"{prefix}_thr{name}_largest_component_soft_mass_cc": 0.0,
        f"{prefix}_thr{name}_largest_component_fraction": 0.0,
    }
    if n_comp <= 0:
        return out
    sizes = np.bincount(comp.ravel())[1:]
    largest_label = int(np.argmax(sizes) + 1)
    largest = comp == largest_label
    largest_volume_cc = float(largest.sum() * voxel_volume_cc)
    total_volume_cc = out[f"{prefix}_thr{name}_binary_volume_cc"]
    out[f"{prefix}_thr{name}_largest_component_volume_cc"] = largest_volume_cc
    out[f"{prefix}_thr{name}_largest_component_soft_mass_cc"] = float(prob[largest].sum() * voxel_volume_cc)
    out[f"{prefix}_thr{name}_largest_component_fraction"] = (
        largest_volume_cc / total_volume_cc if total_volume_cc > 0 else 0.0
    )
    return out


def soft_class_probability_features(prob: np.ndarray, voxel_volume_cc: float, prefix: str) -> dict[str, float]:
    finite = np.nan_to_num(np.asarray(prob, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    soft_mass_cc = float(finite.sum() * voxel_volume_cc)
    out: dict[str, float] = {
        f"{prefix}_soft_volume_cc": soft_mass_cc,
        f"{prefix}_soft_prob_sum": float(finite.sum()),
        f"{prefix}_prob_max": float(finite.max()) if finite.size else 0.0,
        f"{prefix}_prob_mean_all": float(finite.mean()) if finite.size else 0.0,
        f"{prefix}_prob_sq_mass_cc": float(np.square(finite).sum() * voxel_volume_cc),
    }
    out[f"{prefix}_prob_confidence_fraction"] = (
        out[f"{prefix}_prob_sq_mass_cc"] / soft_mass_cc if soft_mass_cc > 0 else 0.0
    )
    out.update(soft_safe_percentiles(finite, prefix))

    for threshold in SOFT_THRESHOLDS:
        name = soft_threshold_name(threshold)
        mask = finite >= threshold
        out[f"{prefix}_prob_thr{name}_binary_volume_cc"] = float(mask.sum() * voxel_volume_cc)
        out[f"{prefix}_prob_thr{name}_soft_mass_cc"] = float(finite[mask].sum() * voxel_volume_cc)
        out[f"{prefix}_prob_thr{name}_mass_fraction"] = (
            out[f"{prefix}_prob_thr{name}_soft_mass_cc"] / soft_mass_cc if soft_mass_cc > 0 else 0.0
        )

    for threshold in SOFT_LITE_COMPONENT_THRESHOLDS.get(prefix, ()):
        out.update(soft_component_features(finite, threshold, voxel_volume_cc, prefix))
    return out


def soft_entropy_features(probabilities: np.ndarray) -> dict[str, float]:
    probs = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-8, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=0)
    fg_prob = np.clip(probs[1] + probs[2], 0.0, 1.0)
    likely_fg = fg_prob >= 0.01
    return {
        "softmax_entropy_mean_all": float(entropy.mean()),
        "softmax_entropy_max": float(entropy.max()),
        "softmax_entropy_mean_fgprob_ge_0p01": float(entropy[likely_fg].mean()) if likely_fg.any() else 0.0,
        "foreground_prob_max": float(fg_prob.max()),
        "foreground_prob_mean_all": float(fg_prob.mean()),
    }


def extract_soft_probability_features(probabilities: np.ndarray | None, reference: sitk.Image) -> dict[str, float]:
    if probabilities is None:
        return {}
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 4 or probs.shape[0] < 3:
        debug_log(f"unexpected probability shape for soft features: {probs.shape}")
        return {}

    spacing_zyx = np.asarray(
        [reference.GetSpacing()[2], reference.GetSpacing()[1], reference.GetSpacing()[0]],
        dtype=np.float64,
    )
    voxel_volume_cc = float(np.prod(spacing_zyx) / 1000.0)
    gtvp = np.nan_to_num(probs[1], nan=0.0, posinf=0.0, neginf=0.0)
    gtvn = np.nan_to_num(probs[2], nan=0.0, posinf=0.0, neginf=0.0)
    foreground = np.clip(gtvp + gtvn, 0.0, 1.0)

    row: dict[str, float] = {
        "prob_shape_z": float(probs.shape[1]),
        "prob_shape_y": float(probs.shape[2]),
        "prob_shape_x": float(probs.shape[3]),
        "prob_spacing_z_mm": float(spacing_zyx[0]),
        "prob_spacing_y_mm": float(spacing_zyx[1]),
        "prob_spacing_x_mm": float(spacing_zyx[2]),
        "prob_voxel_volume_cc": voxel_volume_cc,
    }
    row.update(soft_class_probability_features(gtvp, voxel_volume_cc, "soft_gtvp"))
    row.update(soft_class_probability_features(gtvn, voxel_volume_cc, "soft_gtvn"))
    row.update(soft_class_probability_features(foreground, voxel_volume_cc, "soft_fg"))
    row.update(soft_entropy_features(probs[:3]))

    gtvp_volume = float(row["soft_gtvp_soft_volume_cc"])
    gtvn_volume = float(row["soft_gtvn_soft_volume_cc"])
    fg_volume = float(row["soft_fg_soft_volume_cc"])
    row["soft_gtvn_to_gtvp_volume_ratio"] = gtvn_volume / gtvp_volume if gtvp_volume > 1e-8 else 0.0
    row["soft_gtvp_to_fg_volume_fraction"] = gtvp_volume / fg_volume if fg_volume > 1e-8 else 0.0
    row["soft_gtvn_to_fg_volume_fraction"] = gtvn_volume / fg_volume if fg_volume > 1e-8 else 0.0
    row["soft_gtvp_empty_like"] = float(gtvp_volume < 0.05 or float(row["soft_gtvp_prob_max"]) < 0.05)
    row["soft_gtvn_empty_like"] = float(gtvn_volume < 0.05 or float(row["soft_gtvn_prob_max"]) < 0.05)
    row["soft_fg_empty_like"] = float(fg_volume < 0.05 or float(row["foreground_prob_max"]) < 0.05)
    return row


def normalize_ehr(ehr: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "center": "CenterID",
        "Center": "CenterID",
        "center_id": "CenterID",
        "sex": "Gender",
        "Sex": "Gender",
        "TobaccoConsumption": "Tobacco Consumption",
        "AlcoholConsumption": "Alcohol Consumption",
        "PerformanceStatus": "Performance Status",
        "HPVStatus": "HPV Status",
    }
    out: dict[str, Any] = {}
    for key, value in ehr.items():
        canonical = aliases.get(key, key)
        if canonical in FORBIDDEN_EHR_FIELDS:
            continue
        out[canonical] = normalize_missing(value)
    return out


def normalize_missing(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return np.nan
    return value


def load_model(path: Path) -> tuple[Any, list[str] | None]:
    debug_log(f"loading model: {path}")
    payload = joblib.load(path)
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload["pipeline"], list(payload.get("feature_columns", [])) or None
    return payload, None


def infer_feature_columns(model: Any) -> list[str] | None:
    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
        return None
    columns: list[str] = []
    for _, _, cols in model.named_steps["preprocessor"].transformers:
        if cols == "drop":
            continue
        columns.extend(list(cols))
    return columns


def select_features(row: dict[str, Any], columns: list[str] | None) -> pd.DataFrame:
    if columns is None:
        return pd.DataFrame([row])
    return pd.DataFrame([{col: row.get(col, np.nan) for col in columns}])


def classifier_score(model: Any, x: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        if proba.size:
            return float(np.ravel(proba)[0])
    pred = np.asarray(model.predict(x)).ravel()
    return float(pred[0]) if pred.size else 0.0


def load_t0_rule(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"enabled": False}
    try:
        rule = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        debug_log(f"failed to load T0 rule {path}: {exc!r}")
        return {"enabled": False}
    if not isinstance(rule, dict):
        return {"enabled": False}
    return rule


def is_hpv_positive(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "positive", "pos"}


def finite_feature_value(features: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(features.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def maybe_apply_t0_detector(t_stage: str, n_stage: str, features: dict[str, Any], rule: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    info: dict[str, Any] = {"enabled": bool(rule.get("enabled", False)), "applied": False}
    if not info["enabled"]:
        return t_stage, info
    reasons: list[str] = []
    allowed = {str(item) for item in rule.get("allowed_base_predictions", ["T1"])}
    if str(t_stage) not in allowed:
        reasons.append("base_t_not_allowed")
    if rule.get("hpv_positive_required", True) and not is_hpv_positive(features.get("HPV Status", np.nan)):
        reasons.append("hpv_not_positive")
    gtvp_volume = finite_feature_value(features, "gtvp_volume_cc", np.inf)
    ratio = finite_feature_value(features, "gtvn_to_gtvp_volume_ratio", -np.inf)
    gtvn_volume = finite_feature_value(features, "gtvn_volume_cc", -np.inf)
    if gtvp_volume > float(rule.get("gtvp_volume_max_cc", 3.0)):
        reasons.append("gtvp_not_small")
    if ratio < float(rule.get("gtvn_to_gtvp_volume_ratio_min", 15.0)):
        reasons.append("nodal_ratio_not_high")
    if gtvn_volume < float(rule.get("gtvn_volume_min_cc", 30.0)):
        reasons.append("gtvn_volume_not_high")
    if str(n_stage) not in {str(item) for item in rule.get("n_stage_allowed", ["N2", "N3"])}:
        reasons.append("n_stage_not_high")
    info.update(
        {
            "base_t_stage": str(t_stage),
            "n_stage": str(n_stage),
            "gtvp_volume_cc": gtvp_volume,
            "gtvn_to_gtvp_volume_ratio": ratio,
            "gtvn_volume_cc": gtvn_volume,
            "reasons": reasons,
        }
    )
    if reasons:
        return t_stage, info
    info["applied"] = True
    return "T0", info


class HecktorPipeline:
    def __init__(self, model_root: Path | str = "/opt/ml/model", checkpoint_name: str = "checkpoint_final.pth") -> None:
        self.model_root = Path(model_root)
        debug_log(f"model_root={self.model_root}")
        nnunet_root = self.model_root / "nnunet"
        dataset804_root = nnunet_root / "Dataset804_HE2026_FullOfficial_WideHNROI_Cascade"
        dataset807_root = nnunet_root / "Dataset807_HE2026_FullOfficial_RefineLocalROI_TTACascade_MaskPrior"
        self.lowres_model_dir = dataset804_root / "nnUNetTrainer__nnUNetPlans__3d_lowres"
        self.cascade_model_dir = (
            dataset804_root / "nnUNetTrainer__nnUNetPlans__3d_cascade_fullres"
        )
        self.refinement_model_dir = dataset807_root / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        self.fine_model_dir = self.cascade_model_dir
        self.legacy_dataset803_model_dir = (
            nnunet_root
            / "Dataset803_HE2026_FullOfficial_LocalROI_TinyUNet64Base16Down4AdaptiveZCap"
            / "nnUNetTrainer__nnUNetPlansCoarseSpacingLite24__3d_fullres"
        )
        tabular_root = self.model_root / "tabular"
        self.t_stage_model, self.t_stage_columns = load_model(tabular_root / "t_stage_model.joblib")
        self.n_stage_model, self.n_stage_columns = load_model(tabular_root / "n_stage_model.joblib")
        self.rfs_model, self.rfs_columns = load_model(tabular_root / "rfs_model.joblib")
        self.t0_rule = load_t0_rule(tabular_root / "t0_detector_rule.json")
        self.t_stage_columns = self.t_stage_columns or infer_feature_columns(self.t_stage_model)
        self.n_stage_columns = self.n_stage_columns or infer_feature_columns(self.n_stage_model)
        self.rfs_columns = self.rfs_columns or infer_feature_columns(self.rfs_model)
        self.hn_roi_cfg = HnRoiConfig()
        self.wide_hn_roi_cfg = WideHnRoiConfig()
        self.local_roi_cfg = LocalRoiConfig()
        self.refinement_roi_cfg = RefinementRoiConfig()
        debug_log("initializing Dataset804 3d_lowres nnUNet predictor with checkpoint_final.pth")
        self.lowres_predictor = make_predictor(self.lowres_model_dir, "checkpoint_final.pth")
        debug_log("initializing Dataset804 3d_cascade_fullres nnUNet predictor with checkpoint_final.pth")
        self.cascade_predictor = make_predictor(self.cascade_model_dir, "checkpoint_final.pth")
        debug_log("initializing Dataset807 3d_fullres refinement nnUNet predictor with checkpoint_best.pth")
        self.refinement_predictor = make_predictor(self.refinement_model_dir, "checkpoint_best.pth")
        self.fine_predictor = self.cascade_predictor

    def prepare_case(
        self,
        ct_path: Path,
        pet_path: Path,
        ehr: dict[str, Any],
        output_dir: Path,
        case_id: str | None = None,
    ) -> PreparedCase:
        case_start = time.perf_counter()
        timing: dict[str, float] = {}
        resolved_case_id = case_id or infer_case_id_from_image_path(ct_path)

        stage_start = time.perf_counter()
        debug_log(f"reading ct={ct_path}")
        original_ct = sitk.ReadImage(str(ct_path))
        debug_log(f"reading pet={pet_path}")
        original_pet = sitk.ReadImage(str(pet_path))
        ct_lps = orient_lps(original_ct)
        pet_lps = orient_lps(original_pet)
        timing["read_orient_pet_to_ct_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        initial_hn_bbox, hn_info = compute_hn_roi(ct_lps, pet_lps, self.hn_roi_cfg)
        debug_log(f"initial_hn_bbox={initial_hn_bbox}; method={hn_info.get('roi_method')}")
        initial_hn_ct = crop_image_xyz(ct_lps, initial_hn_bbox)
        if same_grid(ct_lps, pet_lps):
            initial_hn_pet = crop_image_xyz(pet_lps, initial_hn_bbox)
            hn_info["pet_resample_scope"] = "none_ct_pet_same_grid"
        else:
            initial_hn_pet = resample_to_reference(pet_lps, initial_hn_ct, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            hn_info["pet_resample_scope"] = "hn_roi_crop"

        hn_bbox, wide_info = apply_wide_hnroi_z_cap(
            initial_hn_ct,
            initial_hn_pet,
            initial_hn_bbox,
            self.wide_hn_roi_cfg,
        )
        hn_info.update(wide_info)
        debug_log(f"final_hn_bbox={hn_bbox}; wide_action={wide_info.get('wide_hnroi_action')}")
        if hn_bbox == initial_hn_bbox:
            hn_ct = initial_hn_ct
            hn_pet = initial_hn_pet
        else:
            hn_ct = crop_image_xyz(ct_lps, hn_bbox)
            if same_grid(ct_lps, pet_lps):
                hn_pet = crop_image_xyz(pet_lps, hn_bbox)
            else:
                hn_pet = resample_to_reference(pet_lps, hn_ct, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        timing["hn_roi_and_crop_seconds"] = time.perf_counter() - stage_start

        return PreparedCase(
            case_id=resolved_case_id,
            ct_path=ct_path,
            pet_path=pet_path,
            output_dir=output_dir,
            ehr=ehr,
            case_start=case_start,
            timing=timing,
            original_ct=original_ct,
            ct_lps=ct_lps,
            hn_bbox=hn_bbox,
            hn_info=hn_info,
            hn_ct=hn_ct,
            hn_pet=hn_pet,
            tinyunet_input_ncdhw=np.empty((2, 1, 1, 1), dtype=np.float32),
        )

    def build_full_wide_hnroi_case(
        self,
        prepared: PreparedCase,
        previous_stage_seg: np.ndarray | None = None,
    ) -> LocalizedCase:
        nnunet_data, nnunet_properties = image_to_nnunet_array(prepared.hn_ct, prepared.hn_pet)
        size = prepared.hn_ct.GetSize()
        local_bbox_hn = (0, int(size[0]), 0, int(size[1]), 0, int(size[2]))
        local_info = {
            "local_roi_method": "full_wide_hnroi_dataset804_cascade",
            "local_bbox_xyz_hnroi": local_bbox_hn,
            "source": "Dataset804 wide-HNROI; no tumor-centered LocalROI crop",
        }
        return LocalizedCase(
            prepared=prepared,
            local_bbox_hn=local_bbox_hn,
            local_info=local_info,
            local_ct=prepared.hn_ct,
            local_pet=prepared.hn_pet,
            nnunet_data=nnunet_data,
            nnunet_properties=nnunet_properties,
            previous_stage_seg=previous_stage_seg,
        )

    def run_lowres_segmentation(self, prepared: PreparedCase) -> np.ndarray:
        debug_log("running Dataset804 3d_lowres nnUNet")
        stage_start = time.perf_counter()
        localized = self.build_full_wide_hnroi_case(prepared)
        lowres_seg_hn, _ = run_nnunet_prediction_from_array(
            self.lowres_predictor,
            localized.nnunet_data,
            localized.nnunet_properties,
            localized.local_ct,
            return_probabilities=False,
        )
        previous_stage = sitk.GetArrayFromImage(sitk.Cast(lowres_seg_hn, sitk.sitkUInt8)).astype(np.int16)[None]
        prepared.timing["lowres_nnunet_seconds"] = time.perf_counter() - stage_start
        debug_log("Dataset804 3d_lowres completed")
        return previous_stage

    def run_tinyunet_coarse(self, prepared: PreparedCase) -> np.ndarray:
        debug_log("running TinyUNet coarse ensemble")
        stage_start = time.perf_counter()
        coarse_prob = predict_tinyunet_ensemble_prob(
            self.tinyunet_models,
            prepared.tinyunet_input_ncdhw,
            self.tinyunet_device,
        )
        prepared.timing["coarse_tinyunet_seconds"] = time.perf_counter() - stage_start
        debug_log(f"tinyunet_prob shape={coarse_prob.shape}; max={float(np.nanmax(coarse_prob)):.4f}")
        return coarse_prob

    def run_tinyunet_coarse_batch(self, prepared_cases: list[PreparedCase]) -> list[np.ndarray]:
        if not prepared_cases:
            return []
        if len(prepared_cases) == 1:
            return [self.run_tinyunet_coarse(prepared_cases[0])]
        debug_log(f"running TinyUNet coarse batch size={len(prepared_cases)}")
        stage_start = time.perf_counter()
        coarse_probs = predict_tinyunet_ensemble_prob_batch(
            self.tinyunet_models,
            [prepared.tinyunet_input_ncdhw for prepared in prepared_cases],
            self.tinyunet_device,
        )
        elapsed = time.perf_counter() - stage_start
        per_case = elapsed / float(len(prepared_cases))
        for prepared, coarse_prob in zip(prepared_cases, coarse_probs):
            prepared.timing["coarse_tinyunet_seconds"] = per_case
            debug_log(
                f"tinyunet_prob case={prepared.case_id} batch_size={len(prepared_cases)} "
                f"shape={coarse_prob.shape}; max={float(np.nanmax(coarse_prob)):.4f}"
            )
        return coarse_probs

    def build_local_roi(self, prepared: PreparedCase, coarse_prob: np.ndarray) -> LocalizedCase:
        stage_start = time.perf_counter()
        local_bbox_hn, local_info = local_roi_from_tinyunet_probability(coarse_prob, prepared.hn_ct, self.local_roi_cfg)
        debug_log(f"local_bbox_hn={local_bbox_hn}; method={local_info.get('local_roi_method')}")
        local_ct = crop_image_xyz(prepared.hn_ct, local_bbox_hn)
        local_pet = crop_image_xyz(prepared.hn_pet, local_bbox_hn)
        nnunet_data, nnunet_properties = image_to_nnunet_array(local_ct, local_pet)
        debug_log(f"local_ct size={local_ct.GetSize()}; spacing={local_ct.GetSpacing()}")
        prepared.timing["local_roi_and_crop_seconds"] = time.perf_counter() - stage_start
        return LocalizedCase(
            prepared=prepared,
            local_bbox_hn=local_bbox_hn,
            local_info=local_info,
            local_ct=local_ct,
            local_pet=local_pet,
            nnunet_data=nnunet_data,
            nnunet_properties=nnunet_properties,
        )

    def run_fine_segmentation(self, localized: LocalizedCase) -> SegmentedCase:
        debug_log("running Dataset804 3d_cascade_fullres nnUNet")
        stage_start = time.perf_counter()
        fine_seg_local, fine_prob_local = run_nnunet_prediction_from_array(
            self.cascade_predictor,
            localized.nnunet_data,
            localized.nnunet_properties,
            localized.local_ct,
            return_probabilities=True,
            segmentation_previous_stage=localized.previous_stage_seg,
        )
        fine_seg_local = sitk.Cast(fine_seg_local, sitk.sitkUInt8)
        debug_log("Dataset804 3d_cascade_fullres completed")
        localized.prepared.timing["cascade_fullres_nnunet_seconds"] = time.perf_counter() - stage_start
        return SegmentedCase(localized=localized, fine_seg_local=fine_seg_local, fine_prob_local=fine_prob_local)

    def build_refinement_local_roi(self, cascade_segmented: SegmentedCase) -> LocalizedCase:
        stage_start = time.perf_counter()
        cascade_localized = cascade_segmented.localized
        prepared = cascade_localized.prepared
        cascade_hn_seg = cascade_segmented.fine_seg_local
        if cascade_localized.local_bbox_hn != (0, prepared.hn_ct.GetSize()[0], 0, prepared.hn_ct.GetSize()[1], 0, prepared.hn_ct.GetSize()[2]):
            cascade_hn_seg = paste_prediction_to_grid(cascade_hn_seg, prepared.hn_ct, cascade_localized.local_bbox_hn)
        cascade_hn_seg = resample_to_reference(
            cascade_hn_seg,
            prepared.hn_ct,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        local_bbox_hn, local_info = bbox_from_segmentation_with_margin(
            cascade_hn_seg,
            prepared.hn_ct,
            self.refinement_roi_cfg,
        )
        debug_log(f"dataset807_refinement_bbox_hn={local_bbox_hn}; method={local_info.get('local_roi_method')}")
        local_ct = crop_image_xyz(prepared.hn_ct, local_bbox_hn)
        local_pet = crop_image_xyz(prepared.hn_pet, local_bbox_hn)
        local_prior = crop_image_xyz(cascade_hn_seg, local_bbox_hn)
        nnunet_data, nnunet_properties = image_to_nnunet_array_with_priors(local_ct, local_pet, local_prior)
        prepared.timing["dataset807_local_roi_and_crop_seconds"] = time.perf_counter() - stage_start
        return LocalizedCase(
            prepared=prepared,
            local_bbox_hn=local_bbox_hn,
            local_info=local_info,
            local_ct=local_ct,
            local_pet=local_pet,
            nnunet_data=nnunet_data,
            nnunet_properties=nnunet_properties,
            previous_stage_seg=sitk.GetArrayFromImage(sitk.Cast(local_prior, sitk.sitkUInt8)).astype(np.int16)[None],
        )

    def run_refinement_segmentation(self, cascade_segmented: SegmentedCase) -> SegmentedCase:
        cascade_rfs_features = extract_shallow_features(
            cascade_segmented.localized.local_ct,
            cascade_segmented.localized.local_pet,
            cascade_segmented.fine_seg_local,
        )
        cascade_rfs_features.update(
            extract_soft_probability_features(cascade_segmented.fine_prob_local, cascade_segmented.localized.local_ct)
        )
        localized = self.build_refinement_local_roi(cascade_segmented)
        debug_log("running Dataset807 3d_fullres refinement nnUNet")
        stage_start = time.perf_counter()
        refined_seg_local, refined_prob_local = run_nnunet_prediction_from_array(
            self.refinement_predictor,
            localized.nnunet_data,
            localized.nnunet_properties,
            localized.local_ct,
            return_probabilities=True,
        )
        refined_seg_local = sitk.Cast(refined_seg_local, sitk.sitkUInt8)
        prior_arr = localized.previous_stage_seg[0] if localized.previous_stage_seg is not None else None
        refined_arr = sitk.GetArrayFromImage(refined_seg_local).astype(np.uint8, copy=False)
        before_qc = mask_qc_reason(refined_arr, localized.local_ct)
        localized.local_info["empty_tiny_before_reason"] = before_qc["reason"]
        localized.local_info["empty_tiny_before_volume_ml"] = before_qc["total_volume_ml"]
        localized.local_info["empty_tiny_before_largest_component_ml"] = before_qc["largest_component_volume_ml"]
        localized.local_info["refinement_empty_fallback_to_dataset804_prior"] = False
        localized.local_info["segmentation_failed_or_fallback"] = False
        if before_qc["failed"]:
            debug_log(f"Dataset807 refinement empty/tiny ({before_qc['reason']}); trying conservative fallbacks")
            cascade_prob_local = crop_probabilities_by_bbox_xyz(cascade_segmented.fine_prob_local, localized.local_bbox_hn)
            lowres_hn = (
                cascade_segmented.localized.previous_stage_seg[0]
                if cascade_segmented.localized.previous_stage_seg is not None
                else None
            )
            lowres_local = crop_array_by_bbox_xyz(lowres_hn, localized.local_bbox_hn) if lowres_hn is not None else None
            candidates = [
                fallback_candidate_from_probability(cascade_prob_local, localized.local_ct, "cascade_probability_low_threshold"),
                fallback_candidate_from_mask(lowres_local, localized.local_ct, "lowres_hard_mask"),
                fallback_candidate_from_mask(prior_arr, localized.local_ct, "cascade_hard_prior"),
                fallback_candidate_from_probability(refined_prob_local, localized.local_ct, "refinement_probability_low_threshold"),
            ]
            candidate = next((item for item in candidates if item is not None), None)
            if candidate is not None:
                refined_arr = np.asarray(candidate["array"], dtype=np.uint8)
                refined_seg_local = array_to_label_image(refined_arr, localized.local_ct)
                after_qc = mask_qc_reason(refined_arr, localized.local_ct)
                localized.local_info["segmentation_failed_or_fallback"] = True
                localized.local_info["refinement_empty_fallback_to_dataset804_prior"] = True
                localized.local_info["empty_tiny_fallback_source"] = candidate["source"]
                localized.local_info["empty_tiny_fallback_threshold"] = candidate["threshold"]
                localized.local_info["empty_tiny_after_reason"] = after_qc["reason"]
                localized.local_info["empty_tiny_after_volume_ml"] = after_qc["total_volume_ml"]
                localized.local_info["empty_tiny_after_largest_component_ml"] = after_qc["largest_component_volume_ml"]
                debug_log(
                    "empty/tiny fallback selected "
                    f"{candidate['source']} threshold={candidate['threshold']} volume_ml={candidate['total_volume_ml']:.4f}"
                )
            else:
                localized.local_info["empty_tiny_after_reason"] = "no_fallback_candidate"
                localized.local_info["empty_tiny_after_volume_ml"] = before_qc["total_volume_ml"]
                localized.local_info["empty_tiny_after_largest_component_ml"] = before_qc["largest_component_volume_ml"]
        else:
            localized.local_info["empty_tiny_after_reason"] = "ok"
            localized.local_info["empty_tiny_after_volume_ml"] = before_qc["total_volume_ml"]
            localized.local_info["empty_tiny_after_largest_component_ml"] = before_qc["largest_component_volume_ml"]
        debug_log("Dataset807 3d_fullres refinement completed")
        localized.prepared.timing["dataset807_refinement_nnunet_seconds"] = time.perf_counter() - stage_start
        return SegmentedCase(
            localized=localized,
            fine_seg_local=refined_seg_local,
            fine_prob_local=refined_prob_local,
            cascade_rfs_features=cascade_rfs_features,
        )

    def finalize_case(self, segmented: SegmentedCase) -> dict[str, Any]:
        localized = segmented.localized
        prepared = localized.prepared
        prepared.output_dir.mkdir(parents=True, exist_ok=True)

        stage_start = time.perf_counter()
        hn_seg = paste_prediction_to_grid(segmented.fine_seg_local, prepared.hn_ct, localized.local_bbox_hn)
        full_seg_lps = paste_prediction_to_grid(hn_seg, prepared.ct_lps, prepared.hn_bbox)
        full_seg_original = resample_to_reference(
            full_seg_lps,
            prepared.original_ct,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        prepared.timing["paste_to_original_grid_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        hard_features = extract_shallow_features(localized.local_ct, localized.local_pet, segmented.fine_seg_local)
        features = dict(hard_features)
        debug_log("shallow features extracted")
        features.update(extract_soft_probability_features(segmented.fine_prob_local, localized.local_ct))
        debug_log("soft probability features extracted")
        features.update(normalize_ehr(prepared.ehr))
        features["segmentation_failed_or_fallback"] = float(bool(localized.local_info.get("segmentation_failed_or_fallback", False)))

        rfs_features = dict(segmented.cascade_rfs_features or hard_features)
        rfs_features.update(normalize_ehr(prepared.ehr))

        t_stage = str(self.t_stage_model.predict(select_features(features, self.t_stage_columns))[0])
        n_stage = str(self.n_stage_model.predict(select_features(features, self.n_stage_columns))[0])
        t_stage, t0_info = maybe_apply_t0_detector(t_stage, n_stage, features, self.t0_rule)
        rfs_model_risk = classifier_score(self.rfs_model, select_features(rfs_features, self.rfs_columns))
        rfs_model_risk = float(np.clip(rfs_model_risk, 0.0, 1.0))
        # The trained model emits recurrence risk. Grand Challenge C-index implementations often
        # expect larger scores to indicate longer event-free survival, so submit the flipped rank.
        rfs_score = float(1.0 - rfs_model_risk)
        prepared.timing["feature_and_tabular_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        seg_out = prepared.output_dir / "images" / "head-neck-tumor-segmentation"
        seg_out.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(full_seg_original, str(seg_out / "output.mha"), useCompression=True)
        (prepared.output_dir / "t-stage.json").write_text(json.dumps(t_stage), encoding="utf-8")
        (prepared.output_dir / "n-stage.json").write_text(json.dumps(n_stage), encoding="utf-8")
        (prepared.output_dir / "rfs.json").write_text(json.dumps(rfs_score), encoding="utf-8")
        prepared.timing["write_outputs_seconds"] = time.perf_counter() - stage_start
        prepared.timing["total_inside_run_case_seconds"] = time.perf_counter() - prepared.case_start

        return {
            "hn_roi": prepared.hn_info,
            "hn_bbox_xyz_ct_lps": prepared.hn_bbox,
            "local_roi": localized.local_info,
            "t_stage": t_stage,
            "n_stage": n_stage,
            "t0_detector": t0_info,
            "rfs": rfs_score,
            "rfs_model_risk_before_sign_flip": rfs_model_risk,
            "timing": prepared.timing,
        }

    def run_case(self, ct_path: Path, pet_path: Path, ehr: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        prepared = self.prepare_case(ct_path, pet_path, ehr, output_dir)
        previous_stage = self.run_lowres_segmentation(prepared)
        localized = self.build_full_wide_hnroi_case(prepared, previous_stage)
        cascade_segmented = self.run_fine_segmentation(localized)
        refined_segmented = self.run_refinement_segmentation(cascade_segmented)
        return self.finalize_case(refined_segmented)


def locate_single_image(folder: Path) -> Path:
    suffixes = (".mha", ".nii.gz", ".nii", ".nrrd", ".mhd")
    candidates = sorted(p for p in folder.iterdir() if p.is_file() and any(str(p).lower().endswith(s) for s in suffixes))
    if not candidates:
        raise FileNotFoundError(f"No image file found in {folder}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected exactly one image in {folder} for official single-case inference, found {len(candidates)}. "
            "Set HECKTOR_ALLOW_BATCH_ENTRY=1 only for local multi-case benchmarking."
        )
    return candidates[0]


def load_ehr(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected EHR JSON object at {path}")
    return data


def list_image_files(folder: Path) -> list[Path]:
    suffixes = (".mha", ".nii.gz", ".nii", ".nrrd", ".mhd")
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and any(str(p).lower().endswith(s) for s in suffixes))


def infer_case_id_from_image_path(path: Path) -> str:
    name = path.name
    for suffix in ("_0000.nii.gz", "_0001.nii.gz", "__CT.nii.gz", "__PT.nii.gz", ".nii.gz", ".mha", ".nii", ".nrrd", ".mhd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    stem = path.stem
    if stem.endswith("_0000") or stem.endswith("_0001"):
        return stem[:-5]
    if stem.endswith("__CT") or stem.endswith("__PT"):
        return stem[:-4]
    return re.sub(r"(_0000|_0001|__CT|__PT)$", "", stem)


def load_ehr_map(input_dir: Path, case_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not case_ids:
        return {}
    if len(case_ids) == 1 and (input_dir / "ehr.json").exists():
        return {case_ids[0]: load_ehr(input_dir / "ehr.json")}

    mapping: dict[str, dict[str, Any]] = {}
    ehr_dir = input_dir / "ehr"
    if ehr_dir.is_dir():
        for path in sorted(ehr_dir.glob("*.json")):
            mapping[path.stem] = load_ehr(path)

    ehr_csv = input_dir / "ehr.csv"
    if ehr_csv.exists():
        df = pd.read_csv(ehr_csv, dtype=str, keep_default_na=False)
        if "PatientID" in df.columns:
            for _, row in df.iterrows():
                mapping[str(row["PatientID"])] = dict(row)

    return {case_id: mapping.get(case_id, {}) for case_id in case_ids}


def discover_official_case_specs(input_dir: Path, output_dir: Path) -> list[OfficialCaseSpec]:
    ct_files = list_image_files(input_dir / "images" / "ct")
    pet_files = list_image_files(input_dir / "images" / "pet")
    if not ct_files or not pet_files:
        raise FileNotFoundError("Expected at least one CT and one PET image under /input/images/{ct,pet}")

    ct_by_case = {infer_case_id_from_image_path(path): path for path in ct_files}
    pet_by_case = {infer_case_id_from_image_path(path): path for path in pet_files}
    case_ids = sorted(ct_by_case)

    missing_pet = [case_id for case_id in case_ids if case_id not in pet_by_case]
    if missing_pet:
        raise FileNotFoundError(f"Missing PET image for case ids: {missing_pet[:5]}")

    extra_pet = sorted(case_id for case_id in pet_by_case if case_id not in ct_by_case)
    if extra_pet:
        raise FileNotFoundError(f"Missing CT image for PET case ids: {extra_pet[:5]}")

    ehr_map = load_ehr_map(input_dir, case_ids)
    single_case = len(case_ids) == 1
    specs = []
    for case_id in case_ids:
        specs.append(
            OfficialCaseSpec(
                case_id=case_id,
                ct_path=ct_by_case[case_id],
                pet_path=pet_by_case[case_id],
                ehr=ehr_map.get(case_id, {}),
                output_dir=output_dir if single_case else output_dir / case_id,
            )
        )
    return specs


def make_case_result_row(case_id: str, started_at: float, status: str, error: str, debug: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "seconds": time.perf_counter() - started_at,
        "status": status,
        "error": error,
        "timing": debug.get("timing", {}),
    }


def summarize_case_rows(
    rows: list[dict[str, Any]],
    runtime_cfg: BatchRuntimeConfig,
    wall_seconds: float,
    load_seconds: float,
    include_cases: bool = True,
) -> dict[str, Any]:
    ok_seconds = [float(row["seconds"]) for row in rows if row["status"] == "ok"]
    timing_keys = sorted({key for row in rows if row["status"] == "ok" for key in row.get("timing", {})})
    stage_means = (
        {
            key: sum(float(row["timing"].get(key, 0.0)) for row in rows if row["status"] == "ok") / len(ok_seconds)
            for key in timing_keys
        }
        if ok_seconds
        else {}
    )
    summary: dict[str, Any] = {
        "case_count": len(rows),
        "ok_count": len(ok_seconds),
        "error_count": len(rows) - len(ok_seconds),
        "mode": runtime_cfg.mode,
        "cpu_workers": runtime_cfg.cpu_workers,
        "local_roi_workers": runtime_cfg.local_roi_workers,
        "post_workers": runtime_cfg.post_workers,
        "prefetch": runtime_cfg.prefetch,
        "coarse_batch_size": runtime_cfg.coarse_batch_size,
        "load_seconds": load_seconds,
        "benchmark_wall_seconds": wall_seconds,
        "benchmark_wall_seconds_with_load": load_seconds + wall_seconds,
        "throughput_cases_per_minute": (60.0 * len(ok_seconds) / wall_seconds) if ok_seconds and wall_seconds > 0 else None,
        "mean_case_seconds_warm": (sum(ok_seconds) / len(ok_seconds)) if ok_seconds else None,
        "mean_case_seconds_with_load_amortized": ((load_seconds + wall_seconds) / len(ok_seconds)) if ok_seconds else None,
        "stage_means_seconds": stage_means,
    }
    if include_cases:
        summary["cases"] = rows
    return summary


def run_case_specs_sequential(
    pipeline: HecktorPipeline,
    case_specs: list[OfficialCaseSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in case_specs:
        started_at = time.perf_counter()
        status = "ok"
        error = ""
        debug: dict[str, Any] = {}
        try:
            debug = pipeline.run_case(spec.ct_path, spec.pet_path, spec.ehr, spec.output_dir)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        rows.append(make_case_result_row(spec.case_id, started_at, status, error, debug))
    return rows


def run_case_specs_threaded(
    pipeline: HecktorPipeline,
    case_specs: list[OfficialCaseSpec],
    runtime_cfg: BatchRuntimeConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_iter = iter(case_specs)
    pending_prepare: dict[Any, dict[str, Any]] = {}
    pending_localize: dict[Any, dict[str, Any]] = {}
    pending_finalize: dict[Any, dict[str, Any]] = {}
    ready_prepared: deque[tuple[dict[str, Any], PreparedCase]] = deque()
    ready_localized: deque[tuple[dict[str, Any], LocalizedCase]] = deque()

    def inflight_cases() -> int:
        return len(pending_prepare) + len(pending_localize) + len(ready_prepared) + len(ready_localized)

    def submit_next_prepare(executor: ThreadPoolExecutor) -> bool:
        if inflight_cases() >= runtime_cfg.prefetch:
            return False
        try:
            spec = next(case_iter)
        except StopIteration:
            return False
        meta = {"case_id": spec.case_id, "started_at": time.perf_counter()}
        future = executor.submit(pipeline.prepare_case, spec.ct_path, spec.pet_path, spec.ehr, spec.output_dir, spec.case_id)
        pending_prepare[future] = meta
        return True

    def record_error(meta: dict[str, Any], exc: Exception) -> None:
        rows.append(
            make_case_result_row(
                str(meta["case_id"]),
                float(meta["started_at"]),
                "error",
                f"{type(exc).__name__}: {exc}",
                {},
            )
        )

    def drain_ready() -> None:
        for future in list(pending_finalize):
            if future.done():
                meta = pending_finalize.pop(future)
                status = "ok"
                error = ""
                debug: dict[str, Any] = {}
                try:
                    debug = future.result()
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                rows.append(make_case_result_row(str(meta["case_id"]), float(meta["started_at"]), status, error, debug))

        for future in list(pending_localize):
            if future.done():
                meta = pending_localize.pop(future)
                try:
                    ready_localized.append((meta, future.result()))
                except Exception as exc:
                    record_error(meta, exc)

        for future in list(pending_prepare):
            if future.done():
                meta = pending_prepare.pop(future)
                try:
                    ready_prepared.append((meta, future.result()))
                except Exception as exc:
                    record_error(meta, exc)

    with ThreadPoolExecutor(max_workers=runtime_cfg.cpu_workers, thread_name_prefix="hecktor-pre") as pre_executor:
        with ThreadPoolExecutor(max_workers=runtime_cfg.local_roi_workers, thread_name_prefix="hecktor-local") as local_executor:
            post_executor = (
                ThreadPoolExecutor(max_workers=runtime_cfg.post_workers, thread_name_prefix="hecktor-post")
                if runtime_cfg.post_workers > 0
                else None
            )
            try:
                while submit_next_prepare(pre_executor):
                    pass

                while pending_prepare or pending_localize or pending_finalize or ready_prepared or ready_localized:
                    drain_ready()

                    while submit_next_prepare(pre_executor):
                        pass

                    if ready_localized:
                        meta, localized = ready_localized.popleft()
                        try:
                            segmented = pipeline.run_fine_segmentation(localized)
                            if post_executor is None:
                                debug = pipeline.finalize_case(segmented)
                                rows.append(make_case_result_row(str(meta["case_id"]), float(meta["started_at"]), "ok", "", debug))
                            else:
                                pending_finalize[post_executor.submit(pipeline.finalize_case, segmented)] = meta
                        except Exception as exc:
                            record_error(meta, exc)
                        continue

                    if ready_prepared:
                        batch: list[tuple[dict[str, Any], PreparedCase]] = []
                        while ready_prepared and len(batch) < runtime_cfg.coarse_batch_size:
                            batch.append(ready_prepared.popleft())
                        metas = [meta for meta, _ in batch]
                        prepared_cases = [prepared for _, prepared in batch]
                        try:
                            coarse_probs = pipeline.run_tinyunet_coarse_batch(prepared_cases)
                        except Exception:
                            coarse_probs = []
                            for meta, prepared in batch:
                                try:
                                    coarse_probs.append(pipeline.run_tinyunet_coarse(prepared))
                                except Exception as exc:
                                    record_error(meta, exc)
                                    coarse_probs.append(None)
                        for meta, prepared, coarse_prob in zip(metas, prepared_cases, coarse_probs):
                            if coarse_prob is None:
                                continue
                            pending_localize[local_executor.submit(pipeline.build_local_roi, prepared, coarse_prob)] = meta
                        continue

                    wait_set = list(pending_prepare) + list(pending_localize) + list(pending_finalize)
                    if wait_set:
                        wait(wait_set, return_when=FIRST_COMPLETED)

            finally:
                if post_executor is not None:
                    post_executor.shutdown(wait=True, cancel_futures=False)
    return rows


@lru_cache(maxsize=4)
def _resident_pipeline_cached(model_root_str: str, checkpoint_name: str) -> HecktorPipeline:
    debug_log(f"creating resident pipeline for model_root={model_root_str}")
    return HecktorPipeline(model_root=model_root_str, checkpoint_name=checkpoint_name)


def get_resident_pipeline(
    model_root: Path | str = "/opt/ml/model",
    checkpoint_name: str = "checkpoint_final.pth",
) -> HecktorPipeline:
    return _resident_pipeline_cached(str(Path(model_root)), checkpoint_name)


def run_official_io_batch(
    input_dir: Path,
    output_dir: Path,
    model_root: Path | str = "/opt/ml/model",
) -> dict[str, Any]:
    load_start = time.perf_counter()
    pipeline = get_resident_pipeline(model_root=model_root)
    load_seconds = time.perf_counter() - load_start
    case_specs = discover_official_case_specs(input_dir, output_dir)
    runtime_cfg = BatchRuntimeConfig.from_env(len(case_specs))

    if len(case_specs) == 1 or runtime_cfg.mode == "sequential":
        spec = case_specs[0] if len(case_specs) == 1 else None
        if len(case_specs) == 1:
            return pipeline.run_case(spec.ct_path, spec.pet_path, spec.ehr, spec.output_dir)
        wall_start = time.perf_counter()
        rows = run_case_specs_sequential(pipeline, case_specs)
        summary = summarize_case_rows(rows, runtime_cfg, time.perf_counter() - wall_start, load_seconds)
        (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if summary["error_count"] > 0:
            raise RuntimeError(f"Official batch inference had {summary['error_count']} failed cases")
        return {k: v for k, v in summary.items() if k != "cases"}

    wall_start = time.perf_counter()
    rows = run_case_specs_threaded(pipeline, case_specs, runtime_cfg)
    summary = summarize_case_rows(rows, runtime_cfg, time.perf_counter() - wall_start, load_seconds)
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["error_count"] > 0:
        raise RuntimeError(f"Official batch inference had {summary['error_count']} failed cases")
    return {k: v for k, v in summary.items() if k != "cases"}


def run_official_io(input_dir: Path, output_dir: Path, model_root: Path | str = "/opt/ml/model") -> dict[str, Any]:
    if os.environ.get("HECKTOR_ALLOW_BATCH_ENTRY", "").strip() == "1":
        return run_official_io_batch(input_dir, output_dir, model_root=model_root)

    load_start = time.perf_counter()
    pipeline = get_resident_pipeline(model_root=model_root)
    load_seconds = time.perf_counter() - load_start
    ct_path = locate_single_image(input_dir / "images" / "ct")
    pet_path = locate_single_image(input_dir / "images" / "pet")
    ehr = load_ehr(input_dir / "ehr.json")
    debug = pipeline.run_case(ct_path, pet_path, ehr, output_dir)
    debug["entrypoint"] = {
        "mode": "single_case_sequential",
        "batch_entry_enabled": False,
        "pipeline_load_seconds": load_seconds,
    }
    return debug

