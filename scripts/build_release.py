#!/usr/bin/env python3
"""Build both thin nodes on Ubuntu 20.04 and write immutable candidate bundles.

Usage: python3.12 scripts/build_release.py --version 1.0.1 --numpy 1.26.4
No upload is performed. The generated source/OS/dependency receipts accompany
the exact archives used for subsequent PAOS acceptance tests.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--numpy', default='1.26.4')
    parser.add_argument('--kind', choices=('all','benchmark','policy'), default='all')
    parser.add_argument('--index-url', default='https://pypi.tuna.tsinghua.edu.cn/simple')
    args=parser.parse_args()
    if sys.version_info[:2] != (3,12) or platform.machine() != 'x86_64':
        raise SystemExit('Build requires Python 3.12 on Linux x86_64')
    os_release=Path('/etc/os-release').read_text()
    if 'ID=ubuntu' not in os_release or 'VERSION_ID="20.04"' not in os_release:
        raise SystemExit('Release acceptance requires Ubuntu 20.04')
    root=Path(__file__).resolve().parents[1]
    benchmark='calvin' if (root/'adapters/calvin_adapter.py').exists() else 'robotwin'
    build=root/'build'/'release'
    output=root/'dist'/args.version
    build.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    envpath=build/f'python312-numpy{args.numpy}'
    uv=shutil.which('uv')
    if not uv:raise SystemExit('uv is required')
    if not (envpath/'bin/python').exists():run([uv,'venv','--seed','--python',sys.executable,envpath])
    python=envpath/'bin/python'
    requirements=['pyinstaller==6.19.0',f'numpy=={args.numpy}','dora-rs==0.4.1',
                  'forge-msgs==1.2.0','forge-tool==1.0.0','pyarrow==23.0.1',
                  'pydantic==2.13.5','pyyaml==6.0.3','pillow==12.3.0',
                  'websockets==15.0.1','msgpack==1.1.2','msgpack-numpy==0.4.8']
    run([uv,'pip','install','--python',python,'--index-url',args.index_url,*requirements])
    dependencies=subprocess.check_output([uv,'pip','freeze','--python',str(python)],text=True)
    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    source_hashes={}
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    for relative in sorted(set(paths)):
        path=root/relative
        if relative and path.is_file() and not relative.startswith(('build/','dist/')):
            source_hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    for kind in (('benchmark','policy') if args.kind=='all' else (args.kind,)):
        name=f'{benchmark}_{kind}'
        env=os.environ.copy()
        env[f'{benchmark.upper()}_NODE_ROOT']=str(root)
        env[f'{benchmark.upper()}_NODE_KIND']=kind
        run([envpath/'bin/pyinstaller','--noconfirm','--clean','--distpath',build/'bin',
             '--workpath',build/kind,root/'scripts'/f'{benchmark}_benchmark.spec'],env=env,cwd=root)
        executable=build/'bin'/name
        if executable.read_bytes()[:4]!=b'\x7fELF':raise RuntimeError('Expected ELF executable')
        staging=output/f'.{name}.tar.gz.part'
        with staging.open('wb') as out, gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w|') as tar:
                info=tar.gettarinfo(str(executable),arcname=name)
                info.mode=0o755
                info.uid=info.gid=0
                info.uname=info.gname=''
                info.mtime=0
                with executable.open('rb') as data:tar.addfile(info,data)
        sha=hashlib.file_digest(staging.open('rb'),'sha256').hexdigest()
        archive=output/f'{name}-{args.version}-linux-x86_64-{sha}.tar.gz'
        if archive.exists():
            if hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()!=sha:
                raise RuntimeError('Refusing to replace an existing artifact')
            staging.unlink()
        else:staging.rename(archive)
        receipt={'artifact_id':f'{name}-{args.version}-linux-x86_64','version':args.version,
                 'platform':'linux','arch':'x86_64','entrypoint':name,'artifact_type':'executable_tar_gz',
                 'archive':archive.name,'sha256':sha,'size':archive.stat().st_size,
                 'base_commit':source_revision,'source_sha256':source_hashes,
                 'build_os':os_release,'python':sys.version,'dependencies':dependencies,
                 'acceptance_status':'pending'}
        (output/f'{name}.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps({'built':str(archive),'sha256':sha,'acceptance_status':'pending'}),flush=True)


if __name__=='__main__':main()
