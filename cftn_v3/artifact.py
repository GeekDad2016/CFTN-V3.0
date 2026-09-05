"""Atomic self-contained .cftn bundles. Never unpickle arbitrary deployment data."""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from .config import Config, canonical
from .data import file_hash
from .model import CFTN, ByteTokenizer


def save_bundle(path, model, *, training=None, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        root = Path(tmp)
        weights = root/'weights.safetensors'
        save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(weights))
        if not isinstance(model.tokenizer, ByteTokenizer):
            model.tokenizer.save_pretrained(root/'tokenizer')
        if training is not None:
            torch.save(training, root/'training.pt')
        manifest = {'format': 'cftn_v3_bundle_v1', 'config': model.config.as_dict(),
                    'metadata': metadata or {}, 'files': {}}
        for file in root.rglob('*'):
            if file.is_file():
                manifest['files'][file.relative_to(root).as_posix()] = file_hash(file)
        (root/'manifest.json').write_text(canonical(manifest), encoding='utf-8')
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        try:
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as z:
                for file in root.rglob('*'):
                    if file.is_file():
                        z.write(file, file.relative_to(root).as_posix())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return file_hash(path)


def load_bundle(path, device='cpu', training=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or 'manifest.json' not in names:
                raise ValueError('invalid bundle inventory')
            manifest = json.loads(z.read('manifest.json'))
            if manifest.get('format') != 'cftn_v3_bundle_v1':
                raise ValueError('unsupported artifact')
            if set(names) != {'manifest.json', *manifest['files']}:
                raise ValueError('undeclared artifact members')
            for name, expected in manifest['files'].items():
                target = (root/name).resolve()
                if root.resolve() not in target.parents or '\\' in name:
                    raise ValueError('unsafe artifact path')
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, target.open('wb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest)
                if file_hash(target) != expected:
                    raise ValueError('artifact checksum mismatch')
        config = Config(**manifest['config'])
        tokenizer = ByteTokenizer()
        if config.coordinator != 'tiny':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True)
        model = CFTN(config, tokenizer, initialize=False)
        tensors = load_file(str(root/'weights.safetensors'))
        dtype = next(iter(tensors.values())).dtype
        model.to(dtype=dtype)
        model.load_state_dict(tensors, strict=True)
        model.to(device=device).eval()
        state = torch.load(root/'training.pt', map_location='cpu', weights_only=True) if training and (root/'training.pt').exists() else None
        return model, state, manifest['metadata']


def export_bundle(source, destination):
    model, _, metadata = load_bundle(source)
    return save_bundle(destination, model, metadata=metadata)
