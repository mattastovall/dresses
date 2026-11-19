"""Filesystem worker contract: immutable input job -> manifest and review assets."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageOps

from . import pose, pose_inpaint
from .images import canvas_geometry, normalize, png, prepare_canvas, registered_rgb
from .prompts import CHECK_RACK, RACK, RECOLOR
from .proofs import Fabric
from .provider import data_url, digest, locked, run, save_json
from .review import review_page
from .sampling import annotated_rack, distance, parse_vision, sample

EDIT = "ideogram/v4.5/edit"
VISION = "openrouter/router/vision"
SEGMENT = "fal-ai/sam-3/image"


def read(path):
    return json.loads(Path(path).read_text())


def binary(im, size):
    if im.size != size:
        raise ValueError(f"Mask dimensions {im.size} must match {size}")
    raw = (
        im.getchannel("A")
        if "A" in im.getbands() and im.getchannel("A").getextrema() == (0, 255)
        else im.convert("L")
    )
    m = raw.point(lambda x: 255 if x >= 128 else 0)
    if m.getextrema() != (0, 255):
        raise ValueError("Mask must include both foreground and background")
    return m


def edit(cache, source, mask, references, prompt, destination, quality):
    args = {
        "image_url": data_url(source),
        "mask_url": data_url(mask),
        "reference_image_urls": [data_url(f) for f in references],
        "prompt": prompt,
        "edit_precision": "high",
        "quality": quality,
        "image_size": "auto",
        "seed": 39147,
    }
    _, receipt = run(cache, EDIT, args, ("images", destination))
    return receipt


def get_mask(cfg, root, out, cache, worn, jobs):
    if "mask" in cfg:
        with Image.open(root / cfg["mask"]) as supplied:
            if supplied.size != worn.size:
                with Image.open(root / cfg["worn"]) as source:
                    if supplied.size != ImageOps.exif_transpose(source).size:
                        raise ValueError(
                            "Provide an oriented source-size or normalized-size garment mask"
                        )
                supplied = supplied.resize(worn.size, Image.Resampling.NEAREST)
            return binary(supplied, worn.size)
    return segment_parts(cfg, out / "worn.png", worn.size, cache, jobs, "segment")


def segment_parts(cfg, photo, size, cache, jobs, prefix):
    masks = []
    for n, part in enumerate(cfg.get("mask_parts", ["dress"]), 1):
        asset = photo.parent / f"{prefix}-{n}.png"
        body, receipt = run(
            cache,
            SEGMENT,
            {
                "image_url": data_url(photo),
                "prompt": part,
                "apply_mask": False,
                "include_scores": True,
                "return_multiple_masks": True,
                "max_masks": 2,
                "output_format": "png",
            },
            ("masks", asset),
        )
        receipt.update(part=part, scores=body.get("scores"))
        jobs.append(receipt)
        scores = body.get("scores") or []
        if scores and scores[0] is not None and scores[0] < 0.5:
            raise ValueError(
                f"Low mask confidence for {part}. Supply a corrected garment mask."
            )
        masks.append(binary(Image.open(asset), size))
    if not masks:
        raise ValueError("At least one garment mask is required")
    mask = masks[0]
    for part in masks[1:]:
        mask = ImageChops.lighter(mask, part)
    return mask


def collect_colors(cfg, root, out, cache, jobs):
    rows, unresolved = [], []
    for rack_no, filename in enumerate(cfg["racks"], 1):
        prefix = f"rack-{rack_no:02d}"
        im = normalize(root / filename, 1600)
        png(im, out / f"{prefix}.png")
        path = cfg.get("annotations", {}).get(filename)
        if path:
            doc = read(root / path)
        else:
            body, receipt = run(
                cache,
                VISION,
                {
                    "image_urls": [data_url(out / f"{prefix}.png")],
                    "model": cfg.get("vision_model", "google/gemini-3.8-flash"),
                    "temperature": 0,
                    "reasoning": True,
                    "max_tokens": 6000,
                    "prompt": RACK + cfg["description"],
                },
            )
            jobs.append(receipt)
            doc = parse_vision(body["output"])
        if not isinstance(doc.get("observations"), list) or not isinstance(
            doc.get("unresolved"), list
        ):
            raise TypeError(
                "Rack response must contain observations and unresolved arrays"
            )
        save_json(out / f"{prefix}-observations.json", doc)
        unresolved.extend(
            {"rack": prefix, "reason": reason} for reason in doc["unresolved"]
        )
        if not doc["observations"]:
            unresolved.append({"rack": prefix, "reason": "No identifiable dresses"})
        local = []
        for n, o in enumerate(doc["observations"], 1):
            ref_id = f"{prefix}-{n:02d}"
            row, tile = sample(im, o)
            row.update(
                id=ref_id,
                rack=filename,
                source="provided points" if path else "vision locator",
                sku=cfg.get("sku_map", {}).get(ref_id),
                state="sample_review",
                publishable=False,
            )
            if tile is not None:
                row["swatch"] = f"{ref_id}-swatch.png"
                png(tile, out / row["swatch"])
            rows.append(row)
            local.append(row)
        annotated = out / f"{prefix}-annotated.png"
        png(annotated_rack(im, local), annotated)
        if not path and local:
            proposed = [
                {"index": i, "label": r["label"], "points": r["points"]}
                for i, r in enumerate(local, 1)
            ]
            answer, receipt = run(
                cache,
                VISION,
                {
                    "image_urls": [
                        data_url(out / f"{prefix}.png"),
                        data_url(annotated),
                    ],
                    "model": cfg.get("vision_model", "google/gemini-3.8-flash"),
                    "temperature": 0,
                    "reasoning": True,
                    "max_tokens": 4000,
                    "prompt": CHECK_RACK
                    + cfg["description"]
                    + " Proposed locations: "
                    + json.dumps(proposed),
                },
            )
            jobs.append(receipt)
            audit = parse_vision(answer["output"])
            checks = sorted(audit["checks"], key=lambda item: item["index"])
            if [c["index"] for c in checks] != list(
                range(1, len(local) + 1)
            ) or not isinstance(audit["unresolved"], list):
                raise ValueError("Rack audit omitted or duplicated a garment")
            for row, check in zip(local, checks):
                if check["issue"]:
                    row["review_reasons"].append(str(check["issue"]))
            unresolved.extend(
                {"rack": prefix, "reason": s} for s in audit["unresolved"]
            )
            save_json(out / f"{prefix}-audit.json", audit)
    mapping = cfg.get("sku_map", {})
    if (
        any(not isinstance(v, str) or not v.strip() for v in mapping.values())
        or len(set(mapping.values())) != len(mapping)
        or set(mapping) - {r["id"] for r in rows}
    ):
        raise ValueError(
            "sku_map must identify one observed representative per unique SKU"
        )
    if mapping:
        return rows, unresolved
    for i, row in enumerate(rows):
        for other in rows[:i]:
            if (
                "srgb" in row
                and "srgb" in other
                and distance(row["srgb"], other["srgb"]) < 6
            ):
                for a, b in ((row, other), (other, row)):
                    a["review_reasons"].append(
                        "Similar to " + b["id"] + "; confirm SKU mapping"
                    )
    return rows, unresolved


def input_identity(cfg, root):
    paths = list(cfg["racks"]) + [
        cfg[k]
        for k in (
            "worn",
            "detail",
            "mask",
            "model_plate",
            "pose_template",
            "portrait_mask",
        )
        if k in cfg
    ]
    paths += list(cfg.get("annotations", {}).values())
    return {p: digest((root / p).read_bytes()) for p in paths}


def execute(jobfile, out, render=False, only=None):
    jobfile = Path(jobfile).resolve()
    root, out = jobfile.parent, Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = read(jobfile)
    if not cfg.get("style_id") or not cfg.get("racks") or not cfg.get("description"):
        raise ValueError(
            "A style_id, description and at least one rack image are required"
        )
    if not 0 <= cfg.get("headroom_fraction", 0) <= 0.8:
        raise ValueError("headroom_fraction must be 0..0.8")
    if cfg.get("quality", "medium") not in ("very_low", "low", "medium", "high"):
        raise ValueError("Unknown Ideogram quality tier")
    cap = cfg.get("max_color_edits")
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError("max_color_edits must be a nonnegative integer")
    sources = input_identity(cfg, root)
    signature = digest(
        json.dumps(["direct-provider-v1", cfg, sources], sort_keys=True).encode()
    )
    cache = out.parent / ".cache"
    with locked(cache / ("worker-" + digest(str(out).encode()))):
        manifest_file = out / "manifest.json"
        previous = read(manifest_file) if manifest_file.exists() else {}
        if previous and previous["job_sha256"] != signature:
            raise ValueError(
                "Output directory belongs to a different input revision. Select a new directory."
            )
        jobs = []
        m = {
            "schema_version": 1,
            "style_id": cfg["style_id"],
            "job_sha256": signature,
            "input_sha256": sources,
            "render_kind": "provider",
            "state": "preparing",
            "publishable": False,
            "provider_requests": jobs,
        }
        try:
            worn = normalize(root / cfg["worn"], 1536)
            png(worn, out / "worn.png")
            mask_source = get_mask(cfg, root, out, cache, worn, jobs)
            mask_source_fraction = np.asarray(mask_source).mean() / 255
            if not 0.003 < mask_source_fraction < 0.95:
                raise ValueError(
                    "Garment mask coverage is implausible for this worn photo"
                )
            headroom = cfg.get("headroom_fraction", 0)
            isolated, mask = prepare_canvas(worn, mask_source, headroom)
            png(isolated, out / "isolated.png")
            png(ImageOps.invert(mask), out / "inpaint-mask.png")
            size, fitted, offset = canvas_geometry(worn.size, headroom)
            m["registration"] = {
                "canvas_size": size,
                "fitted_size": fitted,
                "offset": offset,
            }
            joints = pose.detect(worn, cache, fitted, offset)
            png(pose.draw_pose(joints, size), out / "pose.png")
            save_json(out / "pose.json", joints)
            m["pose_reference"] = "pose.png"
            m["mask"] = "mask.png"
            rows, unresolved = collect_colors(cfg, root, out, cache, jobs)
            m.update(variants=rows, unresolved=unresolved, rack_files=cfg["racks"])
            if only and only not in {r["id"] for r in rows}:
                raise ValueError("Requested observation ID is absent")
            if "model_plate" in cfg:
                plate = root / cfg["model_plate"]
                m["base_generator"] = cfg.get("plate_provenance", "supplied plate")
            else:
                plate = out / "generated-plate.png"
                template = (
                    root / cfg["pose_template"] if "pose_template" in cfg else None
                )
                if template:
                    with Image.open(template) as registered:
                        if registered.size != size:
                            raise ValueError(
                                "Synthetic pose template must match the registered canvas"
                            )
                strength = cfg.get("base_strength", 0.42 if template else 1.0)
                if not 0 < strength <= 1:
                    raise ValueError(
                        "base_strength must be greater than zero and at most one"
                    )
                jobs.extend(
                    pose_inpaint.generate(
                        cache,
                        out,
                        size,
                        fitted,
                        offset,
                        init=template,
                        strength=strength,
                    )
                )
                m["base_generator"] = pose_inpaint.FLUX
                m["pose_conditioning"] = {
                    "detector": pose_inpaint.DWPOSE,
                    "draw_mode": "body-pose",
                    "control_mode": "pose",
                    "weights": pose_inpaint.UNION,
                    "image": "dwpose.png",
                    "strength": strength,
                    "template": cfg.get("pose_template"),
                }
            base = registered_rgb(plate, size)
            png(base, out / "base.png")
            if cfg.get("portrait_mask"):
                with Image.open(root / cfg["portrait_mask"]) as supplied:
                    mask = binary(supplied, size)
            else:
                mask = segment_parts(
                    cfg, out / "base.png", size, cache, jobs, "portrait-segment"
                )
            png(mask, out / "mask.png")
            png(ImageOps.invert(mask), out / "edit-fabric-mask.png")
            m["base"] = "base.png"
            m["base_sha256"] = digest((out / "base.png").read_bytes())
            m["garment_pixels"] = mask.histogram()[255]
            observed = pose.detect(base, cache, size, (0, 0))
            m["pose_check"] = pose.compare(joints, observed)
            pose.comparison(worn, base, joints, observed, fitted, offset).save(
                out / "pose-comparison.jpg", quality=92
            )
            m["pose_comparison"] = "pose-comparison.jpg"
            fabric = Fabric(base, mask)
            m["base_edge_check"] = fabric.spill_check(mask)
            previous_rows = {r["id"]: r for r in previous.get("variants", [])}
            for row in rows:
                old = previous_rows.get(row["id"], {})
                if "swatch" not in row:
                    continue
                guide = fabric.guide(row["srgb"])
                if guide is not None:
                    row["color_proof"] = row["id"] + "-proof.png"
                    png(guide, out / row["color_proof"])
                if (
                    row["review_reasons"]
                    or unresolved
                    or m["base_edge_check"]["needs_review"]
                    or not m["pose_check"]["within_pilot_tolerance"]
                ):
                    row["state"] = "needs_input_review"
                elif cfg.get("sku_map") and not row["sku"]:
                    row["state"] = "unmapped_observation"
                else:
                    row["state"] = "ready_for_ideogram"
                if (
                    old.get("image")
                    and previous.get("base_sha256") == m["base_sha256"]
                    and (out / old["image"]).exists()
                    and old.get("sha256") == digest((out / old["image"]).read_bytes())
                ):
                    row.update(
                        {
                            k: old[k]
                            for k in ("image", "sha256", "qa", "receipt", "state")
                        }
                    )
            save_json(manifest_file, m)
            rendered = sum("image" in row for row in rows)
            for row in rows:
                if (
                    not render
                    or row["state"] != "ready_for_ideogram"
                    or (only and only != row["id"])
                ):
                    continue
                if cap is not None and rendered >= cap:
                    row["state"] = "edit_limit_reached"
                    continue
                refs = [out / row["swatch"]]
                prompt = RECOLOR.format(label=row["label"], hex=row["hex"])
                if "color_proof" in row:
                    refs.append(out / row["color_proof"])
                    prompt += " Reference image 2 shows the registered color placement. Recover natural lighting and fabric finish while retaining its geometry."
                image_path = out / (row["id"] + "-ideogram.png")
                row["receipt"] = edit(
                    cache,
                    out / "base.png",
                    out / "edit-fabric-mask.png",
                    refs,
                    prompt,
                    image_path,
                    cfg.get("quality", "medium"),
                )
                variant = registered_rgb(image_path, size)
                row.update(
                    image=image_path.name,
                    sha256=digest(image_path.read_bytes()),
                    qa=fabric.audit(variant, row["srgb"]),
                )
                row["state"] = (
                    "needs_detail_review"
                    if row["qa"]["needs_close_inspection"]
                    else "merchant_review"
                )
                rendered += 1
                save_json(manifest_file, m)
            if cfg.get("detail"):
                normalize(root / cfg["detail"], 1600).save(
                    out / "detail-reference.jpg",
                    quality=94,
                    subsampling=0,
                    icc_profile=b"",
                )
                m["detail_reference"] = "detail-reference.jpg"
            review_page(out, m)
            m["state"] = (
                "review_required"
                if render or any(r.get("image") for r in rows)
                else "prepared"
            )
        except Exception as e:
            m.update(state="interrupted", error=str(e)[:1000])
            # Keep receipts for completed variants even if a later prepare stage is interrupted.
            if previous.get("variants") and (
                not m.get("base_sha256")
                or m["base_sha256"] == previous.get("base_sha256")
            ):
                completed = {r["id"]: r for r in previous["variants"] if r.get("image")}
                if "variants" in m:
                    m["variants"] = [
                        completed.get(r["id"], r) if not r.get("image") else r
                        for r in m["variants"]
                    ]
                else:
                    m["variants"] = previous["variants"]
            save_json(manifest_file, m)
            raise
        save_json(manifest_file, m)
        return m
