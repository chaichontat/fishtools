"""Landmark + output-contract helpers for CCF alignment workflows.

Design note:
This module keeps IO concerns (path resolution + JSON load/write) centralized
in `LandmarkRegistrationOutputs` so notebook-style workflows don't
reimplement legacy fallback logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final


PointXY = tuple[float, float]
BBoxYX = tuple[int, int, int, int]  # (r0, r1, c0, c1)


@dataclass(frozen=True, slots=True)
class LandmarkRegistrationOutputs:
    """Output contract produced by `ccf/register_partial_section.py` for a given ROI."""

    root: Path

    P1_LANDMARKS_JSON_NAME: Final[str] = "p1_landmarks.json"
    P1_SIMILARITY_TFM_NAME: Final[str] = "p1_similarity.tfm"
    P1_RESULT_PNG_NAME: Final[str] = "p1_result.png"

    @property
    def p1_landmarks_json(self) -> Path:
        return self.root / self.P1_LANDMARKS_JSON_NAME

    @property
    def p1_similarity_tfm(self) -> Path:
        return self.root / self.P1_SIMILARITY_TFM_NAME

    @property
    def p1_result_png(self) -> Path:
        return self.root / self.P1_RESULT_PNG_NAME

    def try_read_p1_landmarks(self) -> P1Landmarks | None:
        if not self.p1_landmarks_json.exists():
            return None
        return P1Landmarks.from_json(self.p1_landmarks_json)

    def read_p1_landmarks(self) -> P1Landmarks:
        if not self.p1_landmarks_json.exists():
            raise FileNotFoundError(f"Missing p1 landmarks JSON at {self.p1_landmarks_json}")
        return P1Landmarks.from_json(self.p1_landmarks_json)

    def try_read_prior_rotation_deg(self) -> int | None:
        landmarks = self.try_read_p1_landmarks()
        if landmarks is None:
            return None
        return landmarks.prior_rotation_deg

    def write_p1_landmarks_payload(
        self,
        payload: dict[str, object],
        *,
        indent: int = 2,
        encoder: type[json.JSONEncoder] | None = None,
    ) -> None:
        self.p1_landmarks_json.write_text(
            json.dumps(payload, indent=indent, cls=encoder),
            encoding="utf-8",
        )

    def write_p1_landmarks(
        self,
        landmarks: P1Landmarks,
        *,
        indent: int = 2,
        encoder: type[json.JSONEncoder] | None = None,
    ) -> None:
        self.write_p1_landmarks_payload(landmarks.to_json_payload(), indent=indent, encoder=encoder)


@dataclass(frozen=True, slots=True)
class P1Landmarks:
    prior_rotation_deg: int
    fixed_points_cropped_xy: list[PointXY]
    moving_points_fullres_xy_in_rotated_crop: list[PointXY]
    atlas_crop_bbox: BBoxYX
    sample_rotated_crop_bbox: BBoxYX

    atlas_slice_idx: int | None = None
    preview_downsample: int | None = None
    atlas_full_shape_yx: tuple[int, int] | None = None
    sample_rotated_full_shape_yx: tuple[int, int] | None = None

    @staticmethod
    def _parse_bbox(value: object, *, key: str, path: Path) -> BBoxYX:
        if not isinstance(value, list | tuple) or len(value) != 4:
            raise ValueError(f"Invalid {key}={value!r} in {path}; expected 4 integers (r0,r1,c0,c1).")
        try:
            r0, r1, c0, c1 = (int(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key}={value!r} in {path}; expected 4 integers.") from exc
        return (r0, r1, c0, c1)

    @staticmethod
    def _parse_points(value: object, *, key: str, path: Path) -> list[PointXY]:
        if not isinstance(value, list):
            raise ValueError(f"Invalid {key} in {path}; expected a list of [x,y] pairs.")
        points: list[PointXY] = []
        for item in value:
            if not isinstance(item, list | tuple) or len(item) != 2:
                raise ValueError(f"Invalid {key} entry {item!r} in {path}; expected [x,y].")
            try:
                x, y = (float(item[0]), float(item[1]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {key} entry {item!r} in {path}; expected numeric [x,y].") from exc
            points.append((x, y))
        return points

    @staticmethod
    def _parse_shape(value: object, *, key: str, path: Path) -> tuple[int, int] | None:
        if value is None:
            return None
        if not isinstance(value, list | tuple) or len(value) != 2:
            raise ValueError(f"Invalid {key}={value!r} in {path}; expected [y,x] shape.")
        try:
            y, x = (int(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key}={value!r} in {path}; expected 2 integers.") from exc
        return (y, x)

    @classmethod
    def from_json(cls, path: Path) -> P1Landmarks:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid landmarks payload in {path}: expected a JSON object.")

        missing = [
            key
            for key in (
                "prior_rotation_deg",
                "fixed_points_cropped_xy",
                "moving_points_fullres_xy_in_rotated_crop",
                "atlas_crop_bbox",
                "sample_rotated_crop_bbox",
            )
            if key not in payload
        ]
        if missing:
            raise ValueError(f"Missing key(s) {missing!r} in {path}")

        try:
            prior_rotation_deg = int(float(payload["prior_rotation_deg"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid prior_rotation_deg={payload.get('prior_rotation_deg')!r} in {path}") from exc

        atlas_slice_idx_raw = payload.get("atlas_slice_idx")
        if atlas_slice_idx_raw is None:
            atlas_slice_idx = None
        else:
            try:
                atlas_slice_idx = int(atlas_slice_idx_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid atlas_slice_idx={atlas_slice_idx_raw!r} in {path}") from exc

        fixed_points = cls._parse_points(payload["fixed_points_cropped_xy"], key="fixed_points_cropped_xy", path=path)
        moving_points = cls._parse_points(
            payload["moving_points_fullres_xy_in_rotated_crop"],
            key="moving_points_fullres_xy_in_rotated_crop",
            path=path,
        )

        atlas_crop_bbox = cls._parse_bbox(payload["atlas_crop_bbox"], key="atlas_crop_bbox", path=path)
        sample_rotated_crop_bbox = cls._parse_bbox(
            payload["sample_rotated_crop_bbox"], key="sample_rotated_crop_bbox", path=path
        )

        preview_downsample_raw = payload.get("preview_downsample")
        if preview_downsample_raw is None:
            preview_downsample = None
        else:
            try:
                preview_downsample = int(preview_downsample_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid preview_downsample={preview_downsample_raw!r} in {path}") from exc

        return cls(
            prior_rotation_deg=prior_rotation_deg,
            atlas_slice_idx=atlas_slice_idx,
            fixed_points_cropped_xy=fixed_points,
            moving_points_fullres_xy_in_rotated_crop=moving_points,
            atlas_crop_bbox=atlas_crop_bbox,
            sample_rotated_crop_bbox=sample_rotated_crop_bbox,
            preview_downsample=preview_downsample,
            atlas_full_shape_yx=cls._parse_shape(payload.get("atlas_full_shape_yx"), key="atlas_full_shape_yx", path=path),
            sample_rotated_full_shape_yx=cls._parse_shape(
                payload.get("sample_rotated_full_shape_yx"),
                key="sample_rotated_full_shape_yx",
                path=path,
            ),
        )

    def to_json_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "prior_rotation_deg": self.prior_rotation_deg,
            "fixed_points_cropped_xy": [[x, y] for x, y in self.fixed_points_cropped_xy],
            "moving_points_fullres_xy_in_rotated_crop": [[x, y] for x, y in self.moving_points_fullres_xy_in_rotated_crop],
            "atlas_crop_bbox": list(self.atlas_crop_bbox),
            "sample_rotated_crop_bbox": list(self.sample_rotated_crop_bbox),
        }
        if self.atlas_slice_idx is not None:
            payload["atlas_slice_idx"] = self.atlas_slice_idx
        if self.preview_downsample is not None:
            payload["preview_downsample"] = self.preview_downsample
        if self.atlas_full_shape_yx is not None:
            payload["atlas_full_shape_yx"] = list(self.atlas_full_shape_yx)
        if self.sample_rotated_full_shape_yx is not None:
            payload["sample_rotated_full_shape_yx"] = list(self.sample_rotated_full_shape_yx)
        return payload
