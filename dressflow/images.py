import io
import os
import shutil

from PIL import Image, ImageCms, ImageOps

from .provider import atomic

SRGB = ImageCms.createProfile("sRGB")


def normalize(source, longest):
    with Image.open(source) as raw:
        im = ImageOps.exif_transpose(raw)
        profile = im.info.get("icc_profile")
        if profile:
            im = ImageCms.profileToProfile(
                im,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                SRGB,
                outputMode="RGB",
            )
        else:
            im = im.convert("RGB")
        im.thumbnail((longest, longest), Image.Resampling.LANCZOS)
        return im


def png(im, path):
    # Declare sRGB without a time-varying ICC profile creation timestamp in cache keys.
    from PIL.PngImagePlugin import PngInfo

    tags = PngInfo()
    if im.mode == "RGB":
        tags.add(b"sRGB", b"\x00")
    buf = io.BytesIO()
    im.save(buf, format="PNG", pnginfo=tags, icc_profile=None)
    atomic(path, buf.getvalue())


def canvas_geometry(source_size, headroom):
    width, height = source_size
    top = round(height * headroom)
    canvas_height = height + top
    canvas_width = max(width, round(canvas_height * 2 / 3))
    scale = min(1, 1536 / max(canvas_width, canvas_height))
    size = (round(canvas_width * scale), round(canvas_height * scale))
    fitted = (round(width * scale), round(height * scale))
    offset = ((size[0] - fitted[0]) // 2, size[1] - fitted[1])
    return size, fitted, offset


def prepare_canvas(worn, garment, headroom):
    """Use the same transform for the dress, mask and pose reference."""
    if garment.size != worn.size:
        raise ValueError("Mask must match the normalized worn image")
    size, fitted, offset = canvas_geometry(worn.size, headroom)
    mask = Image.new("L", size)
    mask.paste(garment.resize(fitted, Image.Resampling.NEAREST), offset)
    rgb = Image.new("RGB", size, (190, 188, 184))
    rgb.paste(worn.resize(fitted, Image.Resampling.LANCZOS), offset)
    isolated = Image.composite(rgb, Image.new("RGB", size, (190, 188, 184)), mask)
    return isolated, mask


def registered_rgb(path, size):
    with Image.open(path) as im:
        if im.size != size:
            raise ValueError(f"Canvas mismatch for {path}")
        return im.convert("RGB")


def copy_asset(source, target):
    if target.exists() and os.path.samefile(source, target):
        return
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
