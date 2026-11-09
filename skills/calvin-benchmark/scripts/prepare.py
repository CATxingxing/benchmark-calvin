"""Start hook for calvin-benchmark: CALVIN checkout, evaluation dataset and profile weights.

All user settings live in one optional file shared by every benchmark Skill,
~/.PhyAgentOS/benchmarks.yaml. Only the calvin section is read here:

    calvin:
      root: /data/calvin                    # mees/calvin checkout with submodules (needed once)
      dataset: /data/calvin-mtlc-evaluation # evaluation dataset with L6 embeddings (needed once)
      weights_dir: /big/disk/paos-weights   # optional, default ~/.PhyAgentOS/weights/calvin-benchmark
      weights:                              # optional: reuse an existing local copy of a weight
        mcil-d_d-1207: /models/mcil/D_D_static_rgb_baseline

The profile files only use the fixed paths below. This hook links them to the
configured places and checks the dataset files that decide the score against
their lock. It verifies the weights against their lock and downloads and
extracts the official archive when they are missing. It then builds the
private Python environment of each uv-based Node (`<node> --prepare`) so a
first start on a clean machine downloads its packages here, before PAOS starts
the startup_timeout_s clock.

Standard library + PyYAML, Python 3.8+.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PAOS_HOME = Path.home() / ".PhyAgentOS"
CONFIG = PAOS_HOME / "benchmarks.yaml"
SECTION = "calvin"
ROOT_LINK = PAOS_HOME / "data" / "calvin"
DATASET_LINK = PAOS_HOME / "data" / "calvin-dataset"
WEIGHTS_LINK = PAOS_HOME / "weights" / "calvin-benchmark"
STAMPS = PAOS_HOME / "cache" / "weights-verified"
DATASET_LOCK = "locks/calvin_dataset.lock.json"
# Nodes whose launcher builds a uv environment on first start (`--prepare` = build only).
PREPARE_NODES = ("calvin_benchmark", "calvin_policy")
CONFIG_KEYS = {"root", "dataset", "weights_dir", "weights"}
ROOT_REQUIRED = (
    "calvin_env/calvin_env",
    "calvin_env/conf",
    "calvin_env/tacto/tacto",
    "calvin_models/calvin_agent",
    "calvin_models/conf",
)


class Fail(Exception):
    pass


def log(message):
    print("[calvin-benchmark] " + message, file=sys.stderr, flush=True)


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise Fail("PyYAML is missing from the PAOS Python (" + sys.executable + ")")
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def user_config():
    if not CONFIG.is_file():
        return {}
    data = load_yaml(CONFIG) or {}
    if not isinstance(data, dict):
        raise Fail(str(CONFIG) + " must be a mapping")
    section = data.get(SECTION) or {}
    if not isinstance(section, dict):
        raise Fail(str(CONFIG) + ": " + SECTION + " must be a mapping")
    unknown = sorted(set(section) - CONFIG_KEYS)
    if unknown:
        raise Fail(str(CONFIG) + ": unknown keys under " + SECTION + ": " + ", ".join(unknown)
                   + " (allowed: " + ", ".join(sorted(CONFIG_KEYS)) + ")")
    return section


def expand(value):
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).absolute()


def link(fixed, target, key):
    """Point the fixed path at target. Only symlinks are ever replaced."""
    if fixed.is_symlink():
        if Path(os.readlink(str(fixed))) == target:
            return
        fixed.unlink()
    elif fixed.exists():
        raise Fail(str(fixed) + " is a real directory, so " + SECTION + "." + key + " in "
                   + str(CONFIG) + " cannot be linked there. Move it away or drop the setting.")
    fixed.parent.mkdir(parents=True, exist_ok=True)
    fixed.symlink_to(target)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_dir(cfg, key, fixed, example):
    """Link fixed to cfg[key] when set; otherwise require an existing fixed path."""
    if cfg.get(key):
        target = expand(cfg[key])
        if not target.is_dir():
            raise Fail(SECTION + "." + key + " in " + str(CONFIG) + " does not exist: " + str(target))
        link(fixed, target, key)
    elif not fixed.exists():
        raise Fail("CALVIN " + key + " is not configured. Add to " + str(CONFIG) + ":\n\n"
                   "  " + SECTION + ":\n    " + key + ": " + example + "\n\n(see HOST_RUNTIME.md)")
    return fixed.resolve()


def prepare_root(cfg):
    root = configured_dir(cfg, "root", ROOT_LINK, "/path/to/calvin")
    missing = [rel for rel in ROOT_REQUIRED if not (ROOT_LINK / rel).exists()]
    if missing:
        raise Fail("CALVIN root " + str(root) + " is incomplete (submodules?), missing: "
                   + ", ".join(missing))
    log("CALVIN root: " + str(root))


def prepare_dataset(cfg, skill_root):
    dataset = configured_dir(cfg, "dataset", DATASET_LINK, "/path/to/calvin-mtlc-evaluation")
    lock = json.loads((skill_root / DATASET_LOCK).read_text(encoding="utf-8"))
    for item in lock["files"]:
        path = DATASET_LINK / item["path"]
        if not path.is_file():
            raise Fail("CALVIN dataset " + str(dataset) + " is missing " + item["path"])
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise Fail("CALVIN dataset " + str(dataset) + ": " + item["path"] + " does not match "
                       + DATASET_LOCK + " (wrong language embeddings or episode bounds?)")
    if not any((DATASET_LINK / "validation").glob("episode_*.npz")):
        raise Fail("CALVIN dataset " + str(dataset) + " has no validation/episode_*.npz frames")
    log("CALVIN dataset: verified, " + str(dataset))


def find_weight(skill_root, profile):
    entries = load_yaml(skill_root / "weights.yaml") or []
    matches = [entry for entry in entries if entry.get("profile") == profile]
    if not matches:
        log("profile " + profile + " declares no weights")
        return None
    if len(matches) > 1:
        raise Fail("weights.yaml has more than one entry for profile " + profile)
    return matches[0]


def download(url, target, size):
    """Fetch url into target with Range resume. No token is sent; URLs are not logged."""
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise Fail("unsupported download URL scheme: " + url)
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            partial.unlink()
            have = 0
        if have == size:
            break
        request = urllib.request.Request(url, headers={"User-Agent": "paos-calvin-benchmark"})
        if have:
            request.add_header("Range", "bytes=" + str(have) + "-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    have = 0
                mode = "ab" if have else "wb"
                with open(str(partial), mode) as stream:
                    last = time.monotonic()
                    while True:
                        block = response.read(1 << 22)
                        if not block:
                            break
                        stream.write(block)
                        have += len(block)
                        if time.monotonic() - last > 30:
                            log("  %s: %.2f / %.2f GB" % (target.name, have / 1e9, size / 1e9))
                            last = time.monotonic()
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "code", None) or type(error).__name__
            log("  %s: attempt %d failed (%s), retrying" % (target.name, attempt, reason))
            time.sleep(min(30, 2 ** attempt))
    if not partial.exists() or partial.stat().st_size != size:
        raise Fail("download of " + target.name + " did not complete")
    return partial


def fetch_archive(weight_id, lock, target):
    """Download the pinned archive, check it, extract only the locked files into target."""
    archive = lock["archive"]
    cache = STAMPS.parent / "downloads" / (weight_id + ".zip")
    cached = cache.with_name(cache.name + ".partial")
    if cached.is_file() and cached.stat().st_size == archive["size_bytes"]:
        log("weights " + weight_id + ": using the downloaded archive " + str(cached))
    else:
        log("weights " + weight_id + ": downloading the official archive (%.2f GB) from %s"
            % (archive["size_bytes"] / 1e9, urllib.parse.urlsplit(archive["url"]).netloc))
    partial = download(archive["url"], cache, archive["size_bytes"])
    # Plain-HTTP sources are acceptable only because the archive hash is pinned here.
    if sha256(partial) != archive["sha256"]:
        partial.unlink()
        raise Fail("weights " + weight_id + ": archive failed sha256 after download")
    prefix = archive.get("strip_prefix", "")
    with zipfile.ZipFile(str(partial)) as bundle:
        for item in lock["files"]:
            dest = target / item["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".partial")
            with bundle.open(prefix + item["path"]) as source, open(str(tmp), "wb") as sink:
                shutil.copyfileobj(source, sink, 1 << 22)
            if tmp.stat().st_size != item["size_bytes"] or sha256(tmp) != item["sha256"]:
                tmp.unlink()
                raise Fail("weights " + weight_id + ": " + item["path"] + " failed sha256 after extraction")
            os.replace(str(tmp), str(dest))
    partial.unlink()


def prepare_weights(cfg, skill_root, entry):
    weight_id = entry["weight_id"]
    lock = json.loads((skill_root / entry["lock"]).read_text(encoding="utf-8"))
    if lock.get("resource_id") != weight_id:
        raise Fail("lock " + entry["lock"] + " does not belong to " + weight_id)
    if cfg.get("weights_dir"):
        weights_dir = expand(cfg["weights_dir"])
        weights_dir.mkdir(parents=True, exist_ok=True)
        link(WEIGHTS_LINK, weights_dir, "weights_dir")
    WEIGHTS_LINK.mkdir(parents=True, exist_ok=True)
    local = (cfg.get("weights") or {}).get(weight_id)
    target = WEIGHTS_LINK / entry["path"]
    if local:
        local = expand(local)
        if not local.is_dir():
            raise Fail(SECTION + ".weights." + weight_id + " in " + str(CONFIG)
                       + " does not exist: " + str(local))
        link(target, local, "weights." + weight_id)
    files = lock["files"]
    stamp = STAMPS / (weight_id + ".json")
    expected = {item["path"]: [item["size_bytes"], item["sha256"]] for item in files}
    try:
        stamped = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamped = {}

    def current():
        state = {}
        for item in files:
            path = target / item["path"]
            if path.is_file():
                info = path.stat()
                state[item["path"]] = [info.st_size, info.st_mtime_ns]
        return state

    state = current()
    if stamped.get("files") == expected and stamped.get("state") == state and len(state) == len(files):
        log("weights " + weight_id + ": verified, " + str(target.resolve()))
        return
    missing = []
    for item in files:
        path = target / item["path"]
        if path.is_file() and path.stat().st_size == item["size_bytes"]:
            if sha256(path) == item["sha256"]:
                continue
            raise Fail("weights " + weight_id + ": " + item["path"] + " in " + str(target.resolve())
                       + " does not match its lock (sha256). Remove it to download again.")
        missing.append(item["path"])
    if missing and local:
        raise Fail("weights " + weight_id + ": " + ", ".join(missing) + " missing or wrong size "
                   "in the local copy " + str(local))
    if missing:
        target.mkdir(parents=True, exist_ok=True)
        fetch_archive(weight_id, lock, target)
    STAMPS.mkdir(parents=True, exist_ok=True)
    tmp = stamp.with_suffix(".tmp")
    tmp.write_text(json.dumps({"files": expected, "state": current()}), encoding="utf-8")
    os.replace(str(tmp), str(stamp))
    log("weights " + weight_id + ": verified, " + str(target.resolve()))


def prepare_nodes(skill_name, profile):
    """Build Node environments now; PAOS links the profile bin dir before this hook."""
    bin_dir = PAOS_HOME / "forge_runtime" / "environments" / skill_name / profile / "current" / "bin"
    for node in PREPARE_NODES:
        launcher = bin_dir / node
        if not launcher.exists():
            log("skip %s environment prepare: %s not found" % (node, launcher))
            continue
        log("preparing %s environment (first start downloads CUDA torch, several GB)" % node)
        started = time.time()
        if subprocess.call([str(launcher), "--prepare"]) != 0:
            raise Fail("%s --prepare failed" % node)
        log("%s environment ready (%.0fs)" % (node, time.time() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skill-root", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    skill_root = args.skill_root.resolve()
    try:
        cfg = user_config()
        prepare_root(cfg)
        prepare_dataset(cfg, skill_root)
        entry = find_weight(skill_root, args.profile)
        if entry is not None:
            prepare_weights(cfg, skill_root, entry)
        prepare_nodes(os.environ.get("PAOS_SKILL_NAME", "calvin-benchmark"), args.profile)
    except Fail as error:
        log("ERROR: " + str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
