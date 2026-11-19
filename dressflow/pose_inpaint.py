"""Use DWPose body keypoints as the actual FLUX ControlNet conditioning image."""

from PIL import Image

from .images import png
from .prompts import BASE
from .provider import data_url, run

DWPOSE = "fal-ai/dwpose"
FLUX = "fal-ai/flux-general/inpainting"
UNION = "InstantX/FLUX.1-dev-Controlnet-Union"


def generate(cache, out, size, fitted, offset, seed=39147, init=None, strength=1.0):
    _, detector = run(
        cache,
        DWPOSE,
        {"image_url": data_url(out / "worn.png"), "draw_mode": "body-pose"},
        ("image", out / "dwpose-source.png"),
    )
    with (
        Image.open(out / "worn.png") as worn,
        Image.open(out / "dwpose-source.png") as detected,
    ):
        # The detector rounds the long side to its 64px grid (e.g. 512 x 704 for 3:4).
        scale = min(detected.size) / min(worn.size)
        if max(abs(a - b * scale) for a, b in zip(detected.size, worn.size)) > 32:
            raise ValueError("DWPose changed the source aspect ratio")
        control = Image.new("RGB", size)
        control.paste(
            detected.convert("RGB").resize(fitted, Image.Resampling.BILINEAR), offset
        )
        if not control.getbbox():
            raise ValueError("DWPose found no body on the source photograph")
    png(control, out / "dwpose.png")
    # FLUX white edits the surroundings. Ideogram black edits the product.
    args = {
        "image_url": data_url(init or out / "isolated.png"),
        "mask_url": data_url(out / "inpaint-mask.png"),
        "image_size": {"width": size[0], "height": size[1]},
        "prompt": BASE,
        "num_inference_steps": 32,
        "guidance_scale": 3.5,
        "strength": strength,
        "seed": seed,
        "output_format": "png",
        "controlnet_unions": [
            {
                "path": UNION,
                "controls": [
                    {
                        "control_mode": "pose",
                        "control_image_url": data_url(out / "dwpose.png"),
                        "conditioning_scale": 0.9,
                    }
                ],
            }
        ],
    }
    _, generator = run(cache, FLUX, args, ("images", out / "generated-plate.png"))
    return [detector, generator]
