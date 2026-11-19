"""Registered color proofs for a single dyed textile, and checks for Ideogram results."""

import numpy as np
from PIL import Image, ImageFilter

from .sampling import lab


def srgb(labs):
    l, a, b = labs.T
    fy = (l + 16) / 116
    f = np.column_stack([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * [
        0.95047,
        1,
        1.08883,
    ]
    rgb = np.clip(
        xyz
        @ np.array(
            [
                [3.2404542, -0.969266, 0.0556434],
                [-1.5371385, 1.8760108, -0.2040259],
                [-0.4985314, 0.041556, 1.0572252],
            ]
        ),
        0,
        1,
    )
    return np.rint(
        255 * np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    ).astype("uint8")


class Fabric:
    def __init__(self, base, mask):
        self.base = np.array(base)
        self.inside = np.asarray(mask) > 127
        pixels = self.base[self.inside]
        self.values = lab(pixels)
        # Dominant chroma in the garment, independent of how light the folds appear.
        ab = self.values[:, 1:]
        bins = np.floor((ab + 128) / 10).astype(int)
        _, group, counts = np.unique(
            bins, axis=0, return_inverse=True, return_counts=True
        )
        center = np.median(ab[group == counts.argmax()], axis=0)
        self.chroma = np.linalg.norm(center)
        self.direction = center / max(1, self.chroma)
        projection = ab @ self.direction
        strength = projection / max(self.chroma, 1)
        hue_fit = projection / np.maximum(1, np.linalg.norm(ab, axis=1))
        self.weight = np.clip((strength - 0.09) / 0.33, 0, 1) * np.clip(
            (hue_fit - 0.65) / 0.28, 0, 1
        )
        self.diffuse = (strength > 0.60) & (hue_fit > 0.93)
        self.mid = np.median(self.values[self.diffuse, 0]) if self.diffuse.any() else 50

    def spill_check(self, mask):
        """Look for dyed cloth generated beyond the product outline, before paying for variants."""
        if self.chroma < 20:
            return {
                "matching_pixels": None,
                "needs_review": False,
                "method": "visual check required for neutral cloth",
            }
        small = mask.resize(
            (max(1, mask.width // 4), max(1, mask.height // 4)),
            Image.Resampling.NEAREST,
        )
        outer = small.filter(ImageFilter.MaxFilter(15)).resize(
            mask.size, Image.Resampling.NEAREST
        )
        band = (np.asarray(outer) > 0) & (
            np.asarray(mask.filter(ImageFilter.MaxFilter(3))) == 0
        )
        ab = lab(self.base[band])[:, 1:]
        projection = ab @ self.direction
        matched = (projection > self.chroma * 0.5) & (
            projection / np.maximum(1, np.linalg.norm(ab, axis=1)) > 0.96
        )
        count = int(matched.sum())
        return {
            "matching_pixels": count,
            "needs_review": bool(count > max(60, self.inside.sum() * 0.003)),
            "method": "near-edge source-dye similarity",
        }

    def guide(self, color):
        if self.chroma < 15 or self.diffuse.mean() < 0.12:
            return None  # Neutral fabric and neutral beadwork require material review.
        target = lab(color)
        l = self.values[:, 0]
        # Piecewise lightness curve through black, observed midtone and white.
        light = np.interp(l, [0, self.mid, 100], [0, target[0], 100])
        falloff = np.clip(
            1 - 0.5 * np.abs(light - target[0]) / max(target[0], 100 - target[0], 1),
            0.45,
            1,
        )
        dyed = np.column_stack((light, target[1] * falloff, target[2] * falloff))
        w = self.weight[:, None]
        out = self.base.copy()
        out[self.inside] = srgb(self.values * (1 - w) + dyed * w)
        return Image.fromarray(out)

    def audit(self, variant, color):
        out = np.asarray(variant)
        nonproduct_changed = np.any(
            out[~self.inside] != self.base[~self.inside], axis=1
        )
        base_l = Image.fromarray(self.base).convert("L")
        edit_l = variant.convert("L")

        def detail(im):
            return np.asarray(im, dtype=float) - np.asarray(
                im.filter(ImageFilter.GaussianBlur(2)), dtype=float
            )

        a, b = detail(base_l)[self.inside], detail(edit_l)[self.inside]
        active = np.abs(a) > 6
        corr = (
            float(np.corrcoef(a[active], b[active])[0, 1])
            if active.sum() > 10 and np.std(b[active]) > 0 and np.std(a[active]) > 0
            else None
        )
        delta = None
        if self.chroma >= 15 and self.diffuse.mean() >= 0.12:
            output_mid = np.median(lab(out[self.inside][self.diffuse]), axis=0)
            delta = round(float(np.linalg.norm(output_mid - lab(color))), 2)
        return {
            "outside_mask_changed_pixels": int(nonproduct_changed.sum()),
            "detail_correlation": round(corr, 3) if corr is not None else None,
            "midtone_delta_e76_to_rack": delta,
            "needs_close_inspection": corr is None
            or corr < 0.85
            or delta is None
            or delta > 10
            or bool(nonproduct_changed.any()),
        }
