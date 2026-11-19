"""Transfer Ideogram dye and photographed front-panel detail onto one coherent portrait."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import pose
from .images import copy_asset, png, registered_rgb
from .proofs import Fabric, srgb
from .provider import digest, locked, save_json
from .review import review_page
from .sampling import lab


def inverse_map(size, pairs):
    """Fit a smooth inverse warp to reviewed product landmarks in the two images."""
    w, h = size
    points = np.asarray(pairs, float)
    if (
        points.ndim != 3
        or points.shape[1:] != (2, 2)
        or len(points) < 6
        or not np.isfinite(points).all()
    ):
        raise ValueError("Supply at least six source/portrait landmark pairs")
    if (points < 0).any() or (points >= [w, h]).any():
        raise ValueError("Landmarks must lie within the registered canvas")
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    src = np.vstack([points[:, 0], corners]) / h
    dst = np.vstack([points[:, 1], corners]) / h

    def radial(a, b):
        r2 = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
        return r2 * np.log(np.maximum(r2, 1.0e-12))

    poly = np.column_stack((np.ones(len(dst)), dst))
    matrix = np.block(
        [
            [radial(dst, dst) + np.eye(len(dst)) * 1.0e-6, poly],
            [poly.T, np.zeros((3, 3))],
        ]
    )
    coef = np.linalg.solve(matrix, np.vstack([src - dst, np.zeros((3, 2))]))
    ys = np.linspace(0, h - 1, h // 8 + 1)
    xs = np.linspace(0, w - 1, w // 8 + 1)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2) / h
    basis = np.column_stack((radial(grid, dst), np.ones(len(grid)), grid))
    shifts = (basis @ coef * h).reshape(len(ys), len(xs), 2).astype("float32")
    # Interpolate displacements at the same endpoint-aligned mesh coordinates.
    x, y = np.meshgrid(np.arange(w, dtype="float32"), np.arange(h, dtype="float32"))
    delta = cv2.remap(
        shifts,
        x * ((len(xs) - 1) / (w - 1)),
        y * ((len(ys) - 1) / (h - 1)),
        cv2.INTER_LINEAR,
    )
    mx = x + delta[:, :, 0]
    my = y + delta[:, :, 1]
    if (
        np.gradient(mx, axis=1) * np.gradient(my, axis=0)
        - np.gradient(mx, axis=0) * np.gradient(my, axis=1)
    ).min() <= 0:
        raise ValueError("Landmark warp folds over; review the point pairs")
    return mx, my


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def fit_width(portrait, mask, pairs, settings):
    """Adjust photographed width by height, moving the cloth, skin and registration together."""
    if not settings:
        return portrait, mask, pairs
    width, height = portrait.size
    knots = np.asarray(settings, dtype=float)
    if (
        knots.ndim != 2
        or knots.shape[1] != 3
        or len(knots) < 2
        or not np.isfinite(knots).all()
    ):
        raise ValueError(
            "Expected fit rows with height, width multiplier and horizontal offset"
        )
    ys, scales, shifts = knots.T
    if (
        ys[0] != 0
        or ys[-1] != height - 1
        or (np.diff(ys) <= 0).any()
        or (scales <= 0).any()
    ):
        raise ValueError(
            "Width fit must span the canvas with increasing rows and positive scales"
        )
    row = np.arange(height)
    lo = np.clip(np.searchsorted(ys, row, side="right") - 1, 0, len(ys) - 2)
    blend = smooth((row - ys[lo]) / (ys[lo + 1] - ys[lo]))
    scale = scales[lo] + blend * (scales[lo + 1] - scales[lo])
    shift = shifts[lo] + blend * (shifts[lo + 1] - shifts[lo])
    pivot = (width - 1) / 2
    x, y = np.meshgrid(np.arange(width, dtype="float32"), row.astype("float32"))
    source_x = ((x - pivot - shift[:, None]) / scale[:, None] + pivot).astype("float32")

    def warp(im, interp):
        return Image.fromarray(
            cv2.remap(
                np.asarray(im), source_x, y, interp, borderMode=cv2.BORDER_REPLICATE
            )
        )

    fitted_pairs = np.array(pairs, dtype=float, copy=True)
    px, py = fitted_pairs[:, 1].T
    if (py < 0).any() or (py >= height).any():
        raise ValueError("Registration points exceed portrait bounds")
    fitted_pairs[:, 1, 0] = (
        (px - pivot) * np.interp(py, row, scale) + pivot + np.interp(py, row, shift)
    )
    return (
        warp(portrait, cv2.INTER_CUBIC),
        warp(mask, cv2.INTER_LINEAR).point(lambda value: 255 if value > 127 else 0),
        fitted_pairs,
    )


def clean_dye_fringe(portrait, cloth, dye):
    """Use nearby exposed skin or studio color to remove dye spill at the matte edge."""
    rgb = np.array(portrait)
    distance = cv2.distanceTransform((~cloth).astype("uint8"), cv2.DIST_L2, 5)
    band = (distance > 0) & (distance < 4)
    protected = distance >= 5
    _, labels = cv2.distanceTransformWithLabels(
        (~protected).astype("uint8"), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    nearby = np.zeros((int(labels.max()) + 1, 3), dtype="uint8")
    nearby[labels[protected]] = rgb[protected]
    original = lab(rgb[band])
    background = lab(nearby[labels[band]])
    excess = (original[:, 1:] - background[:, 1:]) @ dye
    amount = smooth((excess - 1) / 10) * smooth((4 - distance[band]) / 2)
    original[:, 1:] += amount[:, None] * (background[:, 1:] - original[:, 1:])
    rgb[band] = srgb(original)
    return Image.fromarray(rgb)


def reduce_skin_spill(portrait, cloth, regions):
    """Temper the source dye reflection using nearby skin from the same limb."""
    pixels = np.array(portrait)
    for region in regions:
        x0, y0, x1, y1 = region["area"]
        sample = np.asarray(portrait.crop(region["reference"])).reshape(-1, 3)
        reference = np.median(lab(sample)[:, 1:], axis=0)
        angle = np.arctan2(reference[1], reference[0])
        patch = pixels[y0:y1, x0:x1]
        values = lab(patch.reshape(-1, 3))
        a, b = values[:, 1:].T
        chroma = np.hypot(a, b)
        angle_diff = np.clip(angle - np.arctan2(b, a) - 0.06, 0, 0.65)
        # Feather the correction on exposed skin and preserve luminance texture.
        yy, xx = np.indices(patch.shape[:2])
        edge = np.minimum.reduce([xx, yy, x1 - x0 - 1 - xx, y1 - y0 - 1 - yy])
        strength = smooth(edge.ravel() / 12) * smooth((chroma - 12) / 12)
        strength *= ~cloth[y0:y1, x0:x1].ravel()
        correction = angle_diff * strength
        active = correction > 0
        theta = np.arctan2(b, a) + correction
        values[:, 1:] = chroma[:, None] * np.column_stack(
            (np.cos(theta), np.sin(theta))
        )
        patch[active.reshape(patch.shape[:2])] = srgb(values[active])
    return Image.fromarray(pixels)


class PortraitMaterial:
    def __init__(self, portrait, source, mask, portrait_mask, panel, pairs):
        self.map = inverse_map(portrait.size, pairs)
        self.original = Fabric(source, mask)
        self.original_diffuse = self.original.values[self.original.diffuse]
        mapped = self.warp(np.asarray(mask), cv2.INTER_NEAREST)
        self.inside = np.asarray(portrait_mask) > 127
        self.portrait = np.asarray(
            clean_dye_fringe(portrait, self.inside, self.original.direction)
        )
        self.values = lab(self.portrait[self.inside])
        vec = self.values[:, 1:]
        chroma = np.linalg.norm(vec, axis=1)
        projection = vec @ self.original.direction
        angle_fit = projection / np.maximum(1, chroma)
        core = (mapped[self.inside] > 0) & (angle_fit > 0.93) & (chroma > 30)
        if core.sum() < 1000:
            raise ValueError(
                "Expected a distinct chromatic textile on the retouched portrait"
            )
        direction = np.median(vec[core], axis=0)
        direction /= np.linalg.norm(direction)
        proj = vec @ direction
        hue = proj / np.maximum(1, chroma)
        self.weight = smooth((hue - 0.70) / 0.22) * smooth((proj - 2) / 18)
        self.amount = np.clip(proj / np.median(proj[core]), 0, 1)
        self.shift = np.median(self.original_diffuse[:, 0]) - np.median(
            self.values[core, 0]
        )
        p = np.zeros_like(mapped)
        cv2.fillPoly(p, [np.asarray(panel, np.int32)], 255)
        front = self.warp(cv2.bitwise_and(p, np.asarray(mask)), cv2.INTER_NEAREST) > 0
        overlap = (front & self.inside).astype("uint8")
        distance = cv2.distanceTransform(overlap, cv2.DIST_L2, 5)
        self.panel_weight = smooth((distance - 8) / 9).astype("float32")
        self.source_knots = np.quantile(
            self.original_diffuse[:, 0], np.linspace(0, 1, 15)
        )
        if np.min(np.diff(self.source_knots)) <= 0:
            raise ValueError("Source fabric lacks enough distinct lightness levels")

    def warp(self, pixels, interpolation):
        return cv2.remap(
            pixels, *self.map, interpolation, borderMode=cv2.BORDER_REPLICATE
        )

    def apply(self, rendered, target):
        ref = lab(np.asarray(rendered)[self.original.inside][self.original.diffuse])
        labels = self.original_diffuse[:, 0]
        residual = lab(target) - np.median(ref, axis=0)
        adjustment = residual if 10 < np.linalg.norm(residual) <= 20 else np.zeros(3)
        bins = np.clip(
            np.searchsorted(self.source_knots, labels, side="right") - 1,
            0,
            len(self.source_knots) - 2,
        )
        knots = []
        response = []
        for i in np.unique(bins):
            take = bins == i
            knots.append(np.median(labels[take]))
            response.append(np.median(ref[take], axis=0) + adjustment)
        curve = np.vstack(([0, 0, 0], response, [100, 0, 0]))
        curve[:, 0] = np.maximum.accumulate(curve[:, 0])
        light = np.clip(self.values[:, 0] + self.shift, 0, 100)
        mapped = np.stack(
            [np.interp(light, [0, *knots, 100], col) for col in curve.T], axis=1
        )
        mapped[:, 1:] *= self.amount[:, None]
        new = self.portrait.copy()
        dyed = self.values + (mapped - self.values) * self.weight[:, None]
        # Thin chiffon transmits the background. Keep that light in low-dye fibers
        # while stripping the pink cast even when the target is black or ivory.
        dyed[:, 0] += (self.values[:, 0] - dyed[:, 0]) * (1 - self.amount) ** 1.35
        new[self.inside] = srgb(dyed)
        pixels = np.asarray(rendered)
        if adjustment.any():
            pixels = pixels.copy()
            pixels[self.original.inside] = srgb(
                lab(pixels[self.original.inside])
                + self.original.weight[:, None] * adjustment
            )
        detail = self.warp(pixels, cv2.INTER_CUBIC)
        w = self.panel_weight[:, :, None]
        product = np.rint(new * (1 - w) + detail * w).astype("uint8")
        return Image.fromarray(product), adjustment.tolist()


def retouch(job, out, only=None):
    job = Path(job).resolve()
    cfg = json.loads(job.read_text())
    src = (job.parent / cfg["render_manifest"]).resolve()
    portrait_path = (job.parent / cfg["portrait"]).resolve()
    mask_path = (job.parent / cfg["portrait_mask"]).resolve()
    m = json.loads(src.read_text())
    out = Path(out).resolve()
    root = src.parent
    if m.get("render_kind") != "provider":
        raise ValueError(
            "Use a manifest of direct provider renders for the fabric reference"
        )
    fabric_ref = root / (m.get("fabric_reference") or m["base"])
    if out == root:
        raise ValueError("Use a new revision directory")
    key = digest(
        json.dumps(
            [
                "retouch-v3",
                cfg,
                digest(portrait_path.read_bytes()),
                digest(mask_path.read_bytes()),
                digest(src.read_bytes()),
                digest(fabric_ref.read_bytes()),
                digest((root / m["mask"]).read_bytes()),
            ],
            sort_keys=True,
        ).encode()
    )
    if only and only not in {r["id"] for r in m["variants"]}:
        raise ValueError("Unknown color observation")
    with locked(out.parent / ".cache" / ("worker-" + digest(str(out).encode()))):
        prior = {}
        if (out / "manifest.json").exists():
            previous = json.loads((out / "manifest.json").read_text())
            if previous["job_sha256"] != key:
                raise ValueError("Output belongs to different inputs")
            prior = {r["id"]: r for r in previous["variants"]}
        source = Image.open(fabric_ref).convert("RGB")
        portrait = registered_rgb(portrait_path, source.size)
        with Image.open(mask_path) as im:
            if im.size != source.size:
                raise ValueError("Portrait mask must match the portrait canvas")
            portrait_mask = im.convert("L").point(
                lambda value: 255 if value > 127 else 0
            )
        roi = np.asarray(portrait_mask) > 0
        portrait = reduce_skin_spill(portrait, roi, cfg.get("skin_spill", []))
        portrait, portrait_mask, pairs = fit_width(
            portrait,
            portrait_mask,
            cfg["source_to_portrait_points"],
            cfg.get("horizontal_fit"),
        )
        roi = np.asarray(portrait_mask) > 0
        original_mask = Image.open(root / m["mask"]).convert("L")
        mat = PortraitMaterial(
            portrait,
            source,
            original_mask,
            portrait_mask,
            cfg["source_front_panel"],
            pairs,
        )
        out.mkdir(parents=True, exist_ok=True)
        base = np.rint(
            mat.portrait * (1 - mat.panel_weight[:, :, None])
            + mat.warp(np.asarray(source), cv2.INTER_CUBIC)
            * mat.panel_weight[:, :, None]
        ).astype("uint8")
        png(Image.fromarray(base), out / "base.png")
        png(portrait_mask, out / "mask.png")
        png(
            Image.fromarray(np.rint(255 * mat.panel_weight).astype("uint8")),
            out / "front-detail.png",
        )
        audit = Fabric(Image.fromarray(base), portrait_mask)
        pose_check = None
        joints_path = root / "pose.json"
        if joints_path.is_file():
            observed = pose.detect(
                Image.fromarray(base), out.parent / ".cache", source.size, (0, 0)
            )
            pose_check = pose.compare(json.loads(joints_path.read_text()), observed)
        rows = []
        for row in m["variants"]:
            row = dict(row)
            if row.get("swatch"):
                copy_asset(root / row["swatch"], out / row["swatch"])
            row.pop("color_proof", None)
            if not row.get("image"):
                rows.append(row)
                continue
            if only and only != row["id"]:
                if row["id"] in prior:
                    rows.append(prior[row["id"]])
                else:
                    rows.append(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"image", "sha256", "qa"}
                        }
                        | {"state": "pending_retouch"}
                    )
                continue
            file = root / row["image"]
            if digest(file.read_bytes()) != row["sha256"]:
                raise ValueError("Rendered garment checksum changed")
            im, change = mat.apply(registered_rgb(file, source.size), row["srgb"])
            # Only registered product pixels and color-matched cloth can be modified.
            if np.any(np.asarray(im)[~roi] != mat.portrait[~roi]):
                raise ValueError("Retouch modified a protected pixel")
            row["image"] = row["id"] + "-ideogram.png"
            png(im, out / row["image"])
            row["render_sha256"] = row["sha256"]
            row["sha256"] = digest((out / row["image"]).read_bytes())
            row["qa"] = audit.audit(im, row["srgb"])
            row["dye_correction_lab"] = change
            row["state"] = (
                "needs_detail_review"
                if row["qa"]["needs_close_inspection"]
                else "merchant_review"
            )
            if pose_check and not pose_check["within_pilot_tolerance"]:
                row["state"] = "needs_detail_review"
                row["review_reasons"] = [
                    *row["review_reasons"],
                    "Fitted portrait pose outside pilot tolerance",
                ]
            rows.append(row)
        for i, _ in enumerate(m["rack_files"], 1):
            copy_asset(
                root / f"rack-{i:02d}-annotated.png",
                out / f"rack-{i:02d}-annotated.png",
            )
        if m.get("detail_reference"):
            copy_asset(root / m["detail_reference"], out / "detail-reference.jpg")
        result = {
            "schema_version": 1,
            "style_id": m["style_id"],
            "job_sha256": key,
            "state": "review_required",
            "publishable": False,
            "base": "base.png",
            "mask": "mask.png",
            "retouched_portrait": True,
            "rack_files": m["rack_files"],
            "variants": rows,
            "unresolved": m["unresolved"],
            "source_job_sha256": m["job_sha256"],
            "pose_check": pose_check,
            "portrait_mask_sha256": digest(mask_path.read_bytes()),
            "portrait_sha256": digest(portrait_path.read_bytes()),
            "registered_front_interior_pixels": int((mat.panel_weight == 1).sum()),
            "base_sha256": digest((out / "base.png").read_bytes()),
        }
        save_json(out / "manifest.json", result)
        review_page(out, result)
        return result
