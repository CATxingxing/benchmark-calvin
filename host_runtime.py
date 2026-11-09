"""Expose an explicitly selected Python 3.12 host environment to a thin node."""
from __future__ import annotations
import os
import site as python_site
from pathlib import Path
import sys


def configure_host_paths(variable: str) -> None:
    # Append: frozen protocol dependencies must win over the host's copies.
    for entry in os.environ.get(variable, "").split(os.pathsep):
        if not entry:
            continue
        site = Path(entry).expanduser().resolve()
        if not site.is_dir():
            raise RuntimeError(f"{variable} does not exist: {site}")
        candidates = [site]
        if site.name == "site-packages" and site.parent.parent.name == "lib":
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            if site.parent.name != version:
                raise RuntimeError(f"{variable} must use {version}, got {site.parent.name}")
            candidates += [site.parent, site.parent / "lib-dynload"]
            cfg = site.parent.parent.parent / "pyvenv.cfg"
            if cfg.is_file():
                for line in cfg.read_text(encoding="utf-8").splitlines():
                    key, sep, value = line.partition("=")
                    if sep and key.strip() == "home":
                        stdlib = Path(value.strip()).parent / "lib" / version
                        candidates += [stdlib, stdlib / "lib-dynload"]
                        break
        for candidate in candidates:
            value = str(candidate)
            if candidate.is_dir() and value not in sys.path:
                sys.path.append(value)
        # Respect editable installs in the explicitly selected host environment.
        python_site.addsitedir(str(site))
