"""Durable atomic writes with bounded retries for Windows file-sharing conflicts."""
import os
import sys
import tempfile
import time
from pathlib import Path
from .config import canonical

def replace_with_retry(source,target,attempts=9):
    for attempt in range(attempts):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt+1==attempts:raise
            time.sleep(min(.05*2**attempt,.5))

def atomic_write(path,writer):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            writer(stream);stream.flush();os.fsync(stream.fileno())
        replace_with_retry(name,path)
    finally:
        # Never remove the destination or mask the original write failure.
        try:os.unlink(name)
        except FileNotFoundError:pass
        except OSError:pass

def atomic_json(path,value):
    payload=canonical(value).encode('utf-8')
    atomic_write(path,lambda stream:stream.write(payload))

def read_text_retry(path,attempts=6):
    for attempt in range(attempts):
        try:return Path(path).read_text(encoding='utf-8')
        except PermissionError:
            if attempt+1==attempts:raise
            time.sleep(min(.01*2**attempt,.1))

class StatusPublisher:
    """A display-file sharing conflict must not abort optimizer updates."""
    def __init__(self,path):self.path=Path(path);self.warned=False

    def __call__(self,value):
        try:atomic_json(self.path,value)
        except PermissionError as exc:
            if not self.warned:print(f'Status file temporarily unavailable; training continues: {exc}',file=sys.stderr,flush=True)
            fallback={**value,'status_warning':'Primary status file is locked; showing the fallback status.'}
            try:atomic_json(self.path.with_name('status_fallback.json'),fallback)
            except PermissionError:
                if not self.warned:print('Fallback status file is also locked; see checkpoints and training logs.',file=sys.stderr,flush=True)
            self.warned=True;return False
        if self.warned:print('Primary status file recovered.',file=sys.stderr,flush=True)
        self.warned=False;return True
