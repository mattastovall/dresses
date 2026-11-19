import html

from PIL import Image, ImageDraw, ImageFont

from .provider import atomic


def render_label(row):
    if row.get("render_sha256"):
        return "Retouched Ideogram fabric"
    return "Ideogram 4.5" if row.get("image") else "Local color proof"


def contact_sheet(out, rows):
    shown = [r for r in rows if r.get("image") or r.get("color_proof")]
    if not shown:
        return
    n = min(6, len(shown))
    w, h = 205, 320
    canvas = Image.new("RGB", (n * w, 66 + ((len(shown) + n - 1) // n) * h), "#f5f3ef")
    d = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=25)
    body = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    d.text((16, 15), "Dress color review", font=title, fill="#241f1c")
    for i, row in enumerate(shown):
        x, y = (i % n) * w, (i // n) * h + 62
        with Image.open(out / (row.get("image") or row["color_proof"])) as im:
            im.thumbnail((w - 10, h - 57), Image.Resampling.LANCZOS)
            canvas.paste(im, (x + (w - im.width) // 2, y))
        d.ellipse(
            (x + 10, y + h - 47, x + 25, y + h - 32), fill=row["hex"], outline="#99918d"
        )
        d.text((x + 32, y + h - 50), row["label"], font=body, fill="#241f1c")
        method = render_label(row)
        d.text((x + 32, y + h - 28), method, font=small, fill="#605955")
    canvas.save(out / "colors.jpg", quality=93, subsampling=0)


def review_page(out, manifest):
    rows = manifest["variants"]
    contact_sheet(out, rows)

    def esc(value):
        return html.escape(str(value), quote=True)

    cards = []
    for r in rows:
        file = r.get("image") or r.get("color_proof")
        chip = f'<i style="background:{esc(r["hex"])}"></i>' if "hex" in r else ""
        pic = (
            f'<a href="{esc(file)}"><img loading="lazy" src="{esc(file)}" alt="{esc(r["label"])} portrait"></a>'
            if file
            else ""
        )
        caption = (
            render_label(r)
            if file
            else r.get("state", "Sample review").replace("_", " ")
        )
        notes = list(map(str, r["review_reasons"]))
        if r.get("qa"):
            qa = r["qa"]
            if qa.get("needs_close_inspection"):
                notes.append("Flagged for close inspection")
            notes.append(
                f"Rack ΔE76 {qa['midtone_delta_e76_to_rack']} · Detail correlation {qa['detail_correlation']}"
            )
        reasons = " · ".join(notes)
        cards.append(
            f"<article>{pic}<h3>{chip}{esc(r['label'])}</h3><p>{esc(r.get('hex', ''))} · {caption}</p><p>{esc(reasons)}</p></article>"
        )
    refs = [
        ("Base portrait", "base.png"),
        ("Garment mask", "mask.png"),
        ("Body pose", "pose.png"),
        ("Pose comparison", "pose-comparison.jpg"),
    ]
    if manifest.get("retouched_portrait"):
        refs = [
            ("Retouched base", "base.png"),
            ("Registered fabric detail", "front-detail.png"),
        ]
    if manifest.get("pose_conditioning"):
        refs.append(("DWPose control input", manifest["pose_conditioning"]["image"]))
    if (out / "detail-reference.jpg").exists():
        refs.append(("Original beadwork", "detail-reference.jpg"))
    refs.extend(
        (f"Rack {i + 1} · sample sites", f"rack-{i + 1:02d}-annotated.png")
        for i in range(len(manifest["rack_files"]))
    )
    links = "".join(
        f'<figure><a href="{url}"><img src="{url}" alt="{esc(label)}"></a><figcaption>{esc(label)}</figcaption></figure>'
        for label, url in refs
    )
    generated = sum("image" in r for r in rows)
    output_label = (
        "retouched colorways"
        if manifest.get("retouched_portrait")
        else "Ideogram renders"
    )
    unresolved = "".join(
        f"<p>{esc(item['rack'])} · {esc(item['reason'])}</p>"
        for item in manifest["unresolved"]
    )
    pose_note = ""
    if manifest.get("pose_conditioning"):
        settings = manifest["pose_conditioning"]
        width = manifest["pose_check"]["shoulder_width_ratio"]
        wrist = max(manifest["pose_check"]["displacement_px"].values())
        pose_note = f"<p>DWPose body map · FLUX Inpainting with {esc(settings['control_mode'])} control · shoulder width {width:.3f}× source · largest joint offset {wrist:g} px</p>"
    if manifest.get("retouched_portrait") and manifest.get("pose_check"):
        check = manifest["pose_check"]
        pose_note = f"<p>Detected shoulders measure {check['shoulder_width_ratio']:.3f}× the worn reference. Largest measured joint offset {max(check['displacement_px'].values()):g} px.</p>"
    intro = "Local proofs transfer the measured rack color onto the photographed fabric. Open a portrait at full size to inspect its beadwork and edges."
    if manifest.get("retouched_portrait"):
        intro = "Each colorway combines the registered front of its Ideogram render with the scarf, hands and garment edges of the retouched portrait."
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dress color review</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f7f5f1;color:#231f1c;font:16px/1.45 'DM Sans',system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:32px clamp(16px,4vw,60px) 60px}}
h1,h2{{font-family:'Instrument Serif',Georgia,serif;font-weight:500;line-height:1.1}}h1{{font-size:clamp(38px,6vw,62px);margin:0}}h2{{margin:30px 0 14px;font-size:28px}}
p,figcaption{{color:#5c5551;font-size:14px}}a{{color:inherit}}.refs,.cards{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(155px,100%),1fr))}}
.refs figure{{margin:0}}.refs img{{width:100%;height:180px;object-fit:contain;background:#e9e5df}}.cards{{grid-template-columns:repeat(auto-fill,minmax(min(205px,100%),1fr))}}
article img{{width:100%;height:auto;display:block}}h3{{font-size:17px;font-weight:550;margin:9px 0 0}}i{{width:.85em;height:.85em;display:inline-block;border-radius:50%;margin-right:.5em;box-shadow:inset 0 0 0 1px #0002}}article p{{margin:2px 0 4px}}footer{{margin-top:24px}}
</style><main><h1>Dress color review</h1><p>{esc(manifest["style_id"])} · {len(rows)} rack observations · {generated} {output_label}</p>
<h2>Reference images</h2>{pose_note}<div class="refs">{links}</div><h2>Colorways</h2>
<p>{intro}</p>
{unresolved}<div class="cards">{"".join(cards)}</div><footer><a href="manifest.json">Inventory manifest</a></footer></main></html>"""
    atomic(out / "review.html", page.encode())
