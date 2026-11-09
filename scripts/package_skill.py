"""Repack a reviewed Skill deterministically; reject stale member hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = args.source.resolve()
    files = sorted((p for p in root.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(root).as_posix().casefold())
    assert all(not p.is_symlink() for p in files), "Symlinks are unsupported"
    names = {p.relative_to(root).as_posix() for p in files}
    manifest = json.loads((root / "archive-manifest.json").read_text(encoding="utf-8"))
    assert {x["path"] for x in manifest["files"]} == names - {"archive-manifest.json"}
    for item in manifest["files"]:
        data = (root / item["path"]).read_bytes()
        assert len(data) == item["size"] and hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
    skill = yaml.safe_load((root / "skill.yaml").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{skill['name']}-{skill['version']}.tar.gz"
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
            for path in files:
                info = tar.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                info.mode = 0o755 if path.suffix == ".sh" else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
    if output.exists():
        assert output.read_bytes() == temporary.read_bytes(), "Existing output differs; use a new version/directory"
        temporary.unlink()
    else:
        temporary.replace(output)
    print(json.dumps({"archive": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "size_bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
