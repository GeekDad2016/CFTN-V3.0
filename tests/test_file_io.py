import json
import os
import threading
from pathlib import Path
import pytest
import cftn_v3.file_io as io

def test_atomic_write_retries_sharing_errors(tmp_path,monkeypatch):
    path=tmp_path/'status.json';path.write_text('{"old":true}');replace=os.replace;calls=[]
    def locked_then_ready(source,target):
        calls.append(source)
        if len(calls)<3:raise PermissionError('temporary sharing conflict')
        replace(source,target)
    monkeypatch.setattr(io.os,'replace',locked_then_ready);monkeypatch.setattr(io.time,'sleep',lambda _:None)
    io.atomic_json(path,{'updated':1})
    assert len(calls)==3 and json.loads(path.read_text())=={'updated':1}
    assert list(tmp_path.iterdir())==[path]

def test_read_retries_temporary_access_denial(tmp_path,monkeypatch):
    path=tmp_path/'status.json';path.write_text('{"version":1}');read=Path.read_text;attempts=[]
    def occasionally_locked(p,*args,**kwargs):
        attempts.append(p)
        if len(attempts)<3:raise PermissionError('replacement in progress')
        return read(p,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',occasionally_locked);monkeypatch.setattr(io.time,'sleep',lambda _:None)
    assert json.loads(io.read_text_retry(path))=={'version':1} and len(attempts)==3

def test_failed_checkpoint_publication_preserves_old_file(tmp_path,monkeypatch):
    path=tmp_path/'current.specialist';path.write_bytes(b'last-good-checkpoint')
    def denied(*args):raise PermissionError('persistent lock')
    monkeypatch.setattr(io.os,'replace',denied);monkeypatch.setattr(io.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):io.atomic_write(path,lambda f:f.write(b'new'))
    assert path.read_bytes()==b'last-good-checkpoint'
    assert list(tmp_path.iterdir())==[path]

def test_status_conflict_uses_fallback_and_recovers(tmp_path,monkeypatch):
    path=tmp_path/'status.json';write=io.atomic_json
    def locked(target,value):
        if target==path:raise PermissionError('locked')
        write(target,value)
    monkeypatch.setattr(io,'atomic_json',locked);publish=io.StatusPublisher(path)
    assert not publish({'updated':2,'state':'training'})
    assert json.loads((tmp_path/'status_fallback.json').read_text())['state']=='training'
    monkeypatch.setattr(io,'atomic_json',write)
    assert publish({'updated':3,'state':'training'}) and not publish.warned

def test_dashboard_shows_newest_status_and_fallback_warning(tmp_path):
    from cftn_v3.local_math_dashboard import snapshot
    io.atomic_json(tmp_path/'status.json',{'updated':1,'state':'paused','phase':'old'})
    io.atomic_json(tmp_path/'status_fallback.json',{'updated':2,'state':'paused','phase':'new','status_warning':'Primary status locked'})
    view=snapshot(tmp_path)
    assert view['status']['phase']=='new' and view['errors']==['Primary status locked']
    io.atomic_json(tmp_path/'status.json',{'updated':3,'state':'paused','phase':'recovered'})
    assert snapshot(tmp_path)['status']['phase']=='recovered'

def test_disk_full_is_not_silently_treated_as_a_status_lock(tmp_path,monkeypatch):
    def full(*args):raise OSError(28,'No space left')
    monkeypatch.setattr(io,'atomic_json',full)
    with pytest.raises(OSError,match='No space'):io.StatusPublisher(tmp_path/'status.json')({})

@pytest.mark.skipif(os.name!='nt',reason='Windows sharing semantics')
def test_real_windows_read_lock_retries_until_reader_releases(tmp_path,monkeypatch):
    import ctypes
    from ctypes import wintypes
    path=tmp_path/'status.json';io.atomic_json(path,{'version':1})
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE;kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
    assert handle!=ctypes.c_void_p(-1).value
    blocked=threading.Event();errors=[];replace=os.replace
    def observed(source,target):
        try:replace(source,target)
        except PermissionError:blocked.set();raise
    monkeypatch.setattr(io.os,'replace',observed)
    def publish():
        try:io.atomic_json(path,{'version':2})
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=publish);thread.start()
    try:assert blocked.wait(3),'A real Windows read handle should block replacement'
    finally:kernel.CloseHandle(handle);thread.join(5)
    assert not thread.is_alive() and not errors
    assert json.loads(path.read_text(encoding='utf-8'))=={'version':2}

