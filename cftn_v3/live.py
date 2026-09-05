"""Durable private interaction queue, explicit confirmations, and release ledger."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from .config import identity, canonical, TOWERS, LANGUAGES
from .data import verify, file_hash


class Store:
    def __init__(self, path):
        self.root = Path(path).resolve().parent
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS interactions(id TEXT PRIMARY KEY, payload TEXT NOT NULL,
            created REAL NOT NULL, verified INTEGER NOT NULL, consumed TEXT, superseded INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, value TEXT NOT NULL, stable INTEGER NOT NULL,
            created REAL NOT NULL, supersedes TEXT);
          CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY, path TEXT NOT NULL, checksum TEXT NOT NULL,
            created REAL NOT NULL, report TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,prompt TEXT,response TEXT,language TEXT,release TEXT,created REAL);
        ''')

    def ingest(self, row, *, confirm=False):
        r = dict(row)
        if r.get('tower') not in TOWERS or r.get('language') not in LANGUAGES:
            raise ValueError('unsupported capability or language')
        if not r.get('prompt') or not r.get('target'):
            raise ValueError('interaction needs input and proposed target')
        # Public ingestion never trusts caller-supplied human confirmation claims.
        if r.get('source') != 'cftn_bounded_bilingual_v1':
            r['source'] = 'private_interaction'
            r['verifier'] = 'unverified'
            r.pop('confirmation_id', None)
        key = identity({k: r.get(k) for k in ('tower', 'language', 'prompt', 'target')})
        if confirm:
            r['confirmation_id'] = key
            r['verifier'] = 'human_confirmed_v1'
        r['id'] = key
        r.setdefault('criterion', 'private_correction')
        r['verified'] = verify(r)
        if r.get('knowledge_kind') == 'fact' and not r.get('stable_fact'):
            r['verified'] = False
        with self.db:
            self.db.execute('INSERT INTO interactions(id,payload,created,verified) VALUES(?,?,?,?) '
                'ON CONFLICT(id) DO UPDATE SET payload=CASE WHEN excluded.verified>verified THEN excluded.payload ELSE payload END, '
                'verified=MAX(verified,excluded.verified)', (key, canonical(r), time.time(), int(r['verified'])))
            if r.get('supersedes'):
                self.db.execute('UPDATE interactions SET superseded=1 WHERE id=?', (r['supersedes'],))
        if r.get('knowledge_kind') == 'fact':
            self.fact(key, canonical({'prompt': r['prompt'], 'value': r['target']}), r.get('stable_fact', False), r.get('supersedes'))
        return {'id': key, 'verified': r['verified']}

    def event(self, prompt, response, language, release):
        key = identity([prompt, response, language, release, time.time_ns()])
        with self.db:
            self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)', (key, prompt, response, language, release, time.time()))
        return key

    def feedback(self, event_id, tower, target, **fields):
        event = self.db.execute('SELECT prompt,language FROM events WHERE id=?', (event_id,)).fetchone()
        if not event:
            raise ValueError('unknown interaction')
        return self.ingest({'prompt': event[0], 'language': event[1], 'tower': tower,
                            'target': target, **fields}, confirm=True)

    def memory(self, query, limit=4):
        words = {w.casefold() for w in query.split() if len(w)>3}
        rows = self.db.execute('SELECT id,value FROM facts WHERE id NOT IN '
            '(SELECT supersedes FROM facts WHERE supersedes IS NOT NULL) ORDER BY created DESC').fetchall()
        ranked = sorted(rows, key=lambda r: sum(w in r[1].casefold() for w in words), reverse=True)
        return [r[1] for r in ranked if any(w in r[1].casefold() for w in words)][:limit]

    def pending(self, tower, force=False, now=None):
        now = now or time.time()
        rows = [(json.loads(p), created) for p, created in self.db.execute(
            'SELECT payload,created FROM interactions WHERE verified=1 AND consumed IS NULL AND superseded=0')]
        rows = [(r, t) for r, t in rows if r['tower'] == tower]
        due = len(rows) >= 128 or len(rows) >= 32 and (force or now-min(t for _, t in rows) >= 86400)
        return [r for r, _ in rows] if due else []

    def fact(self, key, value, stable=False, supersedes=None):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO facts VALUES(?,?,?,?,?)',
                            (key, value, int(stable), time.time(), supersedes))
            if supersedes:
                self.db.execute('UPDATE interactions SET superseded=1 WHERE id=?', (supersedes,))

    def activate(self, bundle, report, consumed=()):
        if not report.get('passed'):
            raise ValueError('candidate failed release gates')
        path = str(Path(bundle).resolve())
        checksum = file_hash(path)
        release = checksum[:24]
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO releases VALUES(?,?,?,?,?)',
                            (release, path, checksum, time.time(), canonical(report)))
            self.db.execute("INSERT OR REPLACE INTO settings VALUES('active',?)", (release,))
            for key in consumed:
                self.db.execute('UPDATE interactions SET consumed=? WHERE id=? AND consumed IS NULL', (release, key))
        stale = self.db.execute('SELECT id,path FROM releases ORDER BY created DESC LIMIT -1 OFFSET 3').fetchall()
        for old_id, old_path in stale:
            old = Path(old_path).resolve()
            # Only retire artifacts owned directly by this release store.
            if old_id != release and old.parent == self.root and old.suffix == '.cftn':
                old.unlink(missing_ok=True)
                with self.db:
                    self.db.execute('DELETE FROM releases WHERE id=?', (old_id,))
        return release

    def active(self):
        row = self.db.execute("SELECT r.id,r.path,r.checksum FROM releases r JOIN settings s ON r.id=s.value WHERE s.key='active'").fetchone()
        if row and file_hash(row[1]) != row[2]:
            raise ValueError('accepted artifact was modified')
        return {'id': row[0], 'path': row[1]} if row else None

    def rollback(self):
        current = self.active()
        row = self.db.execute('SELECT id,path,checksum FROM releases WHERE id!=? ORDER BY created DESC LIMIT 1',
                              ((current or {}).get('id',''),)).fetchone()
        if not row or file_hash(row[1]) != row[2]:
            raise ValueError('no valid previous release')
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES('active',?)", (row[0],))
        return row[0]


class GPULock:
    """One process owns inference/training GPU work; stale locks are explicit errors."""
    def __init__(self, root):
        self.path = Path(root)/'gpu.lock'

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise RuntimeError('GPU job already active; queue this job until the lock is released')
        os.write(self.fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *args):
        os.close(self.fd)
        self.path.unlink(missing_ok=True)
