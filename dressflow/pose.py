"""Body-only pose reference and a scale/position check in the same image frame."""

import math

import httpx
import numpy as np
from PIL import Image, ImageDraw

from .provider import atomic, digest

MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
JOINTS = {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27}


def detect(image, cache, fitted, offset):
    import mediapipe as mp

    path = cache / "pose-full.task"
    if not path.exists():
        response = httpx.get(MODEL, timeout=90)
        response.raise_for_status()
        if digest(response.content) != SHA256:
            raise ValueError("Pose Landmarker model checksum mismatch")
        atomic(path, response.content)
    if digest(path.read_bytes()) != SHA256:
        raise ValueError(f"Invalid local model {path}")
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)), num_poses=2
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.asarray(image.convert("RGB"))
            )
        )
    if len(result.pose_landmarks) != 1:
        raise ValueError(
            "Pose review required: expected one detected person in worn view"
        )
    landmarks = result.pose_landmarks[0]
    # Cropped faces can flip anatomical L/R. Use the shoulder's image side for BOTH limbs.
    left_index = int(landmarks[11].x > landmarks[12].x)
    points = {}
    for side, idx in (("image_left", left_index), ("image_right", 1 - left_index)):
        for name, start in JOINTS.items():
            p = landmarks[start + idx]
            if (
                min(p.visibility, p.presence) >= 0.50
                and 0 <= p.x <= 1
                and 0 <= p.y <= 1
            ):
                points[f"{side}_{name}"] = [
                    round(offset[0] + p.x * (fitted[0] - 1)),
                    round(offset[1] + p.y * (fitted[1] - 1)),
                ]
    if not all(f"{s}_shoulder" in points for s in ("image_left", "image_right")):
        raise ValueError(
            "Pose review required: shoulders could not be located reliably"
        )
    return points


def draw_pose(points, size):
    im = Image.new("RGB", size, "black")
    d = ImageDraw.Draw(im)

    def bone(a, b, color):
        if a in points and b in points:
            d.line(
                (tuple(points[a]), tuple(points[b])),
                fill=color,
                width=max(3, size[0] // 110),
            )

    for joint in ("shoulder", "hip"):
        bone("image_left_" + joint, "image_right_" + joint, "white")
    for side, color in (("image_left", "#ff944d"), ("image_right", "#47b8ed")):
        for a, b in (
            ("shoulder", "elbow"),
            ("elbow", "wrist"),
            ("shoulder", "hip"),
            ("hip", "knee"),
            ("knee", "ankle"),
        ):
            bone(side + "_" + a, side + "_" + b, color)
    for x, y in points.values():
        r = max(3, size[0] // 140)
        d.ellipse((x - r, y - r, x + r, y + r), fill="white")
    return im


def compare(target, observed):
    def width(p):
        return math.dist(p["image_left_shoulder"], p["image_right_shoulder"])

    common = sorted(target.keys() & observed.keys())
    w = width(target)
    if w < 1:
        raise ValueError("Degenerate shoulder landmarks")
    offsets = {k: round(math.dist(target[k], observed[k]), 1) for k in common}
    scale = width(observed) / w
    missing = sorted(target.keys() - observed.keys())
    passed = (
        abs(scale - 1) <= 0.12 and not missing and max(offsets.values()) / w <= 0.26
    )
    return {
        "shoulder_width_ratio": round(scale, 3),
        "displacement_px": offsets,
        "missing_joints": missing,
        "within_pilot_tolerance": passed,
    }


def comparison(worn, base, target, observed, fitted, offset):
    """Show both subjects on the same canvas and the measurable registration error."""
    from PIL import ImageFont

    source = Image.new("RGB", base.size, "#ccc9c4")
    source.paste(worn.resize(fitted, Image.Resampling.LANCZOS), offset)
    overlay = Image.blend(base, Image.new("RGB", base.size, "#262626"), 0.48)
    d = ImageDraw.Draw(overlay)
    for points, color in ((target, "#4fe6ea"), (observed, "#ffa64b")):
        for side in ("image_left", "image_right"):
            for a, b in (
                ("shoulder", "elbow"),
                ("elbow", "wrist"),
                ("shoulder", "hip"),
                ("hip", "knee"),
                ("knee", "ankle"),
            ):
                k = side + "_" + a
                j = side + "_" + b
                if k in points and j in points:
                    d.line((tuple(points[k]), tuple(points[j])), fill=color, width=5)
        for x, y in points.values():
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
    w = 330
    h = round(base.height * w / base.width)
    out = Image.new("RGB", (3 * w, h + 60), "#f5f3ef")
    cap = ImageDraw.Draw(out)
    for i, (im, label) in enumerate(
        (
            (source, "Worn reference"),
            (base, "Replacement model"),
            (overlay, "Body landmarks"),
        )
    ):
        out.paste(im.resize((w, h), Image.Resampling.LANCZOS), (i * w, 0))
        cap.text(
            (i * w + 10, h + 6),
            label,
            fill="#272322",
            font=ImageFont.load_default(size=17),
        )
    cap.text(
        (2 * w + 10, h + 30),
        "Cyan: source   Orange: replacement",
        fill="#514942",
        font=ImageFont.load_default(size=14),
    )
    return out
