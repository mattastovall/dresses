import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def lab(rgb):
    """sRGB (D65) -> CIELAB for descriptive photo-to-photo distances."""
    v = np.asarray(rgb, dtype=float) / 255
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    xyz = (
        linear
        @ np.array(
            [
                [0.4124564, 0.2126729, 0.0193339],
                [0.3575761, 0.7151522, 0.1191920],
                [0.1804375, 0.0721750, 0.9503041],
            ]
        )
        / [0.95047, 1, 1.08883]
    )
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [
            116 * f[..., 1] - 16,
            500 * (f[..., 0] - f[..., 1]),
            200 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def distance(a, b):
    return float(np.linalg.norm(lab(a) - lab(b)))


def parse_vision(text):
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return json.loads("\n".join(lines))


def sample(im, observation):
    label = observation.get("label")
    if not isinstance(label, str) or not label.strip() or len(label) > 100:
        raise ValueError("Each rack observation needs a short color label")
    points = observation.get("points", [])
    warnings = (
        [str(observation["review_reason"])] if observation.get("review_reason") else []
    )
    record = {"label": label, "points": points, "review_reasons": warnings}
    if len(points) != 3 or any(
        not isinstance(p, list)
        or len(p) != 2
        or any(type(n) is not int or not 0 <= n <= 1000 for n in p)
        for p in points
    ):
        warnings.append("Three in-bounds fabric locations are required")
        return record, None
    radius = max(2, round(min(im.size) * 0.004))
    centers, colors, tiles = [], [], []
    for x, y in points:
        cx, cy = round((im.width - 1) * x / 1000), round((im.height - 1) * y / 1000)
        if any(math.dist((cx, cy), center) <= 2 * radius for center in centers):
            warnings.append("Sample patches overlap")
        centers.append((cx, cy))
        tile = im.crop(
            (
                max(0, cx - radius),
                max(0, cy - radius),
                min(im.width, cx + radius + 1),
                min(im.height, cy + radius + 1),
            )
        )
        pixels = np.array(tile).reshape(-1, 3)
        light = lab(pixels)[:, 0]
        # Median diffuse pixel color after dropping bright stones and deep fold shadows.
        lo, hi = np.percentile(light, [20, 80])
        colors.append(np.median(pixels[(light >= lo) & (light <= hi)], axis=0))
        tiles.append(tile)
    # Choose an observed patch medoid, avoiding a synthetic midpoint between two garments.
    d = np.linalg.norm(lab(colors)[:, None] - lab(colors)[None, :], axis=-1)
    rgb = np.rint(colors[d.sum(axis=1).argmin()]).astype(int).tolist()
    record.update(
        srgb=rgb,
        hex="#" + "".join(f"{c:02X}" for c in rgb),
        patch_rgb=np.rint(colors).astype(int).tolist(),
        patch_radius_px=radius,
        max_patch_delta_e76=round(float(d.max()), 2),
    )
    if d.max() > 16:
        warnings.append("Fabric patches disagree under this lighting")
    ref = Image.new("RGB", (256, 256), tuple(rgb))
    for i, tile in enumerate(tiles):
        ref.paste(tile.resize((85, 85), Image.Resampling.NEAREST), (i * 85, 171))
    return record, ref


def annotated_rack(im, observations):
    out = im.copy()
    d = ImageDraw.Draw(out)
    r = max(5, im.width // 150)
    font = ImageFont.load_default(size=max(14, im.width // 52))
    for i, o in enumerate(observations, 1):
        if "srgb" not in o:
            continue
        for x, y in o["points"]:
            cx = round((im.width - 1) * x / 1000)
            cy = round((im.height - 1) * y / 1000)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline="black", width=4)
            d.ellipse(
                (cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2),
                outline="white",
                width=2,
            )
            d.text(
                (cx, cy - 2 * r),
                str(i),
                font=font,
                anchor="ms",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
    return out
