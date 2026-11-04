"""下载固定版本官方 smol-smoltalk，流式写盘并校验 SHA-256；不训练。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent / "data/smol-smoltalk"
REPO = "HuggingFaceTB/smol-smoltalk"
REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(item):
    name = item["rfilename"]
    path = ROOT / "raw" / Path(name).name
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
    expected = item.get("lfs", {}).get("sha256")
    if path.exists():
        if path.stat().st_size != item["size"] or (expected and digest(path) != expected):
            raise ValueError(f"Existing file does not match source; refusing to overwrite: {path}")
    else:
        partial = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                count = 0
                with urllib.request.urlopen(url, timeout=45) as response, partial.open("wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                        count += len(chunk)
                        if count % (32 * 1024 * 1024) == 0:
                            print(f"{path.name}: {count / 1e6:.0f}/{item['size'] / 1e6:.0f} MB", flush=True)
                if count != item["size"] or (expected and digest(partial) != expected):
                    raise ValueError(f"Download checksum/size mismatch: {name}")
                partial.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
    result = dict(file=f"raw/{path.name}", bytes=path.stat().st_size, sha256=digest(path), url=url)
    print(f"Verified: {path.name}", flush=True)
    return result


def main():
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    assert metadata["sha"] == REVISION
    files = [x for x in metadata["siblings"] if x["rfilename"].endswith(".parquet") or x["rfilename"] == "README.md"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download, files))
    manifest = dict(repo=REPO, revision=REVISION, license="Apache-2.0", files=results)
    (ROOT / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Download complete; all source checksums verified.", flush=True)


if __name__ == "__main__":
    main()
