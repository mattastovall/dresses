"""Durable fal queue requests. A lost submit acknowledgement requires reconciliation."""

import base64
import fcntl
import hashlib
import json
import mimetypes
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def digest(data):
    return hashlib.sha256(data).hexdigest()


def data_url(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def save_json(path, value):
    atomic(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        yield


def hosted_inputs(value, cache):
    """Keep queue requests small. Cache identity stays tied to the original media bytes."""
    if isinstance(value, list):
        return [hosted_inputs(item, cache) for item in value]
    if isinstance(value, dict):
        return {k: hosted_inputs(v, cache) for k, v in value.items()}
    if not isinstance(value, str) or not value.startswith("data:image/"):
        return value
    import fal_client

    header, encoded = value.split(",", 1)
    mime = header[5:].split(";")[0]
    data = base64.b64decode(encoded, validate=True)
    key = digest(data)
    folder = cache / "uploads" / key
    with locked(folder / "lock"):
        receipt = folder / "upload.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if time.time() - previous["time"] < 24 * 3600:
                return previous["url"]
        url = fal_client.upload(
            data, mime, file_name=key[:16] + (mimetypes.guess_extension(mime) or ".png")
        )
        save_json(receipt, {"url": url, "time": time.time()})
        return url


def run(cache, endpoint, args, asset=None):
    """Persist receipts before polling. Cache identity includes all input bytes and options."""
    key = digest(json.dumps([endpoint, args], sort_keys=True).encode())
    folder = cache / key
    folder.mkdir(parents=True, exist_ok=True)
    with locked(folder / "lock"):
        receipt = folder / "request.json"
        result = folder / "result.json"
        if receipt.exists():
            job = json.loads(receipt.read_text())
        else:
            job = None
        if job and not job.get("request_id"):
            raise RuntimeError(
                f"Submit acknowledgement missing. Reconcile fal history with {receipt} before retrying."
            )
        if not result.exists():
            with httpx.Client(
                headers={"Authorization": "Key " + os.environ["FAL_KEY"]}, timeout=90
            ) as client:
                if job is None:
                    submitted = hosted_inputs(args, cache)
                    job = {
                        "endpoint": endpoint,
                        "input_sha256": key,
                        "started_at": time.time(),
                        "submitted_input": submitted,
                    }
                    save_json(receipt, job)
                    response = client.post(
                        "https://queue.fal.run/" + endpoint, json=submitted
                    )
                    if 400 <= response.status_code < 500:
                        receipt.unlink()  # synchronous rejection, no accepted job
                        raise RuntimeError(
                            f"{endpoint} submission rejected ({response.status_code}): {response.text[:300]}"
                        )
                    response.raise_for_status()
                    job.update(response.json())
                    save_json(receipt, job)
                print(f"{endpoint} {job['request_id']}", flush=True)
                for url in (job["status_url"], job["response_url"]):
                    if (
                        urlsplit(url).netloc != "queue.fal.run"
                        or urlsplit(url).scheme != "https"
                    ):
                        raise ValueError("Unexpected fal queue origin")
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    response = client.get(job["status_url"])
                    response.raise_for_status()
                    state = response.json()["status"]
                    if state == "COMPLETED":
                        break
                    if state not in ("IN_QUEUE", "IN_PROGRESS"):
                        raise RuntimeError(
                            f"fal returned {state} for {job['request_id']}"
                        )
                    time.sleep(1.5)
                else:
                    raise TimeoutError(f"Rerun to resume {job['request_id']}")
                response = client.get(job["response_url"])
                if response.is_error:
                    raise RuntimeError(
                        f"{endpoint} {job['request_id']}: HTTP {response.status_code} {response.text[:420]}"
                    )
                save_json(
                    result,
                    {
                        "body": response.json(),
                        "billable_units": response.headers.get("x-fal-billable-units"),
                    },
                )
        payload = json.loads(result.read_text())
        info = {
            "endpoint": endpoint,
            "request_id": job["request_id"],
            "billable_units": payload["billable_units"],
        }
        if asset:
            field, destination = asset
            local = folder / "asset"
            if not local.exists():
                items = payload["body"][field]
                if isinstance(items, dict):
                    items = [items]
                if len(items) != 1:
                    raise ValueError(
                        f"Expected one {field} result in {result}, got {len(items)}"
                    )
                response = httpx.get(items[0]["url"], timeout=90, follow_redirects=True)
                response.raise_for_status()
                atomic(local, response.content)
            atomic(destination, local.read_bytes())
        return payload["body"], info
