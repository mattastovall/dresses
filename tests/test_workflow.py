import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image, ImageDraw

from dressflow import provider, workflow
from dressflow.images import png
from dressflow.sampling import parse_vision, sample


class WorkerTests(unittest.TestCase):
    def test_sampling_missing_overlapping_and_divergent_fabric(self):
        rack = Image.new("RGB", (600, 400), "red")
        ImageDraw.Draw(rack).rectangle((300, 0, 600, 400), fill="blue")
        for pts in (
            [[30, 30], [30, 30], [30, 30]],
            [[2000, 1]],
            [[100, 300], [200, 500], [700, 300]],
        ):
            r, _ = sample(rack, {"label": "test", "points": pts})
            self.assertTrue(r["review_reasons"])
        r, _ = sample(
            rack, {"label": "blue", "points": [[600, 800], [750, 600], [800, 100]]}
        )
        self.assertEqual("#0000FF", r["hex"])
        self.assertFalse(r["review_reasons"])
        self.assertEqual({"test": 1}, parse_vision('```json\n{"test":1}\n```'))

    def test_complete_worker_redelivery_and_pose_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = Image.new("RGB", (320, 480), "#cccccc")
            ImageDraw.Draw(im).rectangle((120, 120, 210, 460), fill="#e23776")
            im.save(root / "worn.png")
            rack = Image.new("RGB", (400, 250), "#123e84")
            ImageDraw.Draw(rack).rectangle((200, 0, 400, 250), fill="#e6a62e")
            rack.save(root / "rack.png")
            cfg = {
                "style_id": "SKU-test",
                "description": "single color",
                "worn": "worn.png",
                "racks": ["rack.png"],
            }
            provider.save_json(root / "job.json", cfg)
            calls = []

            def decode(uri):
                return Image.open(io.BytesIO(base64.b64decode(uri.split(",")[1])))

            def run(cache, endpoint, args, asset=None):
                calls.append((endpoint, args))
                receipt = {"endpoint": endpoint, "request_id": str(len(calls))}
                if endpoint == workflow.SEGMENT:
                    m = Image.new("L", im.size)
                    left = 140 if asset[1].name.startswith("portrait-segment") else 120
                    ImageDraw.Draw(m).rectangle((left, 120, 210, 460), fill=255)
                    png(m, asset[1])
                    return {"scores": [0.94]}, receipt
                if endpoint == workflow.VISION:
                    if len(args["image_urls"]) == 1:
                        answer = {
                            "observations": [
                                {
                                    "label": "blue",
                                    "points": [[100, 200], [200, 500], [350, 800]],
                                },
                                {
                                    "label": "gold",
                                    "points": [[600, 200], [700, 500], [800, 800]],
                                },
                            ],
                            "unresolved": [],
                        }
                    else:
                        answer = {
                            "checks": [
                                {"index": 1, "issue": ""},
                                {"index": 2, "issue": "partially obscured"},
                            ],
                            "unresolved": [],
                        }
                    return {"output": json.dumps(answer)}, receipt
                if endpoint == workflow.pose_inpaint.DWPOSE:
                    self.assertEqual("body-pose", args["draw_mode"])
                    control = Image.new("RGB", im.size)
                    ImageDraw.Draw(control).line(
                        (185, 80, 200, 330), fill="green", width=5
                    )
                    png(control, asset[1])
                    return {}, receipt
                m = np.array(decode(args["mask_url"]))
                # Both these API masks are black over the garment. FLUX edits white;
                # Ideogram edits black. Each stage operates on the opposite region.
                self.assertEqual(0, m[200, 160])
                self.assertEqual(255, m[5, 5])
                if endpoint == workflow.pose_inpaint.FLUX:
                    condition = args["controlnet_unions"][0]["controls"][0]
                    self.assertEqual("pose", condition["control_mode"])
                    self.assertEqual(
                        (320, 480), decode(condition["control_image_url"]).size
                    )
                    self.assertEqual(
                        (0, 128, 0),
                        decode(condition["control_image_url"]).getpixel((185, 80)),
                    )
                    self.assertEqual(0, m[200, 130])
                    generated = Image.new("RGB", (320, 480), "#684872")
                    ImageDraw.Draw(generated).rectangle(
                        (140, 120, 210, 460), fill="#e85593"
                    )
                else:
                    self.assertEqual(workflow.EDIT, endpoint)
                    self.assertEqual(2, len(args["reference_image_urls"]))
                    self.assertNotIn("controlnet_unions", args)
                    self.assertEqual(255, m[200, 130])
                    generated = Image.new("RGB", (320, 480), "#684872")
                    ImageDraw.Draw(generated).rectangle(
                        (140, 120, 210, 460), fill="#123e84"
                    )
                    generated.putpixel((130, 200), (19, 21, 23))
                png(generated, asset[1])
                return {}, receipt

            pose = {
                "image_left_shoulder": [90, 120],
                "image_right_shoulder": [220, 120],
            }
            out = root / "output"
            with (
                patch.object(workflow, "run", run),
                patch.object(workflow.pose_inpaint, "run", run),
                patch.object(workflow.pose, "detect", return_value=pose),
            ):
                m = workflow.execute(root / "job.json", out, render=True)
                first, second = m["variants"]
                self.assertEqual("needs_input_review", second["state"])
                self.assertEqual(
                    (104, 72, 114), Image.open(out / "base.png").getpixel((130, 200))
                )
                self.assertEqual(
                    (19, 21, 23), Image.open(out / first["image"]).getpixel((130, 200))
                )
                self.assertEqual("needs_detail_review", first["state"])
                self.assertEqual(1, first["qa"]["outside_mask_changed_pixels"])
                self.assertFalse(first["publishable"])
                self.assertTrue((out / first["image"]).exists())
                self.assertEqual(
                    first["sha256"],
                    workflow.execute(root / "job.json", out, render=True)["variants"][
                        0
                    ]["sha256"],
                )
                self.assertEqual(
                    1,
                    sum(
                        1
                        for endpoint, a in calls
                        if endpoint == workflow.EDIT
                        and len(a["reference_image_urls"]) == 2
                    ),
                )
                with self.assertRaisesRegex(ValueError, "different input revision"):
                    cfg["style_id"] = "different"
                    provider.save_json(root / "job.json", cfg)
                    workflow.execute(root / "job.json", out)

    def test_queue_resumes_after_status_failure(self):
        posts = []
        fail = [True]

        def request(req):
            if req.method == "POST":
                posts.append(req)
                return httpx.Response(
                    200,
                    json={
                        "request_id": "same",
                        "status_url": "https://queue.fal.run/test/status",
                        "response_url": "https://queue.fal.run/test/result",
                    },
                )
            if req.url.path.endswith("status"):
                if fail.pop() if fail else False:
                    return httpx.Response(503)
                return httpx.Response(200, json={"status": "COMPLETED"})
            return httpx.Response(200, json={"output": "ok"})

        client = httpx.Client
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("os.environ", FAL_KEY="test"),
            patch.object(
                provider.httpx,
                "Client",
                side_effect=lambda **kwargs: client(
                    transport=httpx.MockTransport(request), **kwargs
                ),
            ),
        ):
            for i in range(3):
                if i == 0:
                    with self.assertRaises(httpx.HTTPStatusError):
                        provider.run(Path(tmp), "test", {"prompt": "same"})
                else:
                    body, receipt = provider.run(Path(tmp), "test", {"prompt": "same"})
                    self.assertEqual("ok", body["output"])
                    self.assertEqual("same", receipt["request_id"])
            self.assertEqual(1, len(posts))


if __name__ == "__main__":
    unittest.main()
