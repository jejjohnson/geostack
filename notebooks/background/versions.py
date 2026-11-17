# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Background · versions and identity: hashes, snapshots, commits, DVC with Icechunk
#
# Regenerates every measured number on
# `docs/design/pages/versions-identity.html` (stub: `docs/design/versions_identity.md`).
# Hashes a band, counts the chunks two versions share, commits both to an Icechunk
# repository (tags, a discarded session, two writers in conflict), then runs the
# DVC-plus-Icechunk pointer-file pattern end to end (`dvc_icechunk.json`).
#
# Needs: NumPy, zarr, icechunk, PyYAML, and the `git` and `dvc` commands.
# It illustrates a docs page and is not a stage of the DVC pipeline; run it by hand:
#
#     python notebooks/background/versions.py
#
# It writes `docs/images/background_versions/results.json`, the file the page's
# numbers were taken from. Snapshot ids are generated and differ on every run. About two minutes.

# %%
"""Identity and snapshots: hashes, dedupe, Icechunk commits."""
import hashlib, json, os, time, pathlib, shutil, tempfile
import numpy as np, zarr, icechunk

# %%
# Where the recorded output goes. Set BACKGROUND_OUT to write somewhere else.
import os as _os
import pathlib as _pathlib

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(_os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_versions"))
OUT.mkdir(parents=True, exist_ok=True)

R = {}
rng = np.random.default_rng(0)
band = rng.integers(0, 4000, (5490, 5490), dtype=np.uint16)
raw = band.tobytes()
t = time.perf_counter(); h1 = hashlib.sha256(raw).hexdigest(); R["sha256_band_ms"] = round((time.perf_counter() - t) * 1e3, 1)
R["band_mb"] = round(len(raw) / 1e6, 2); R["h1"] = h1
b2 = band.copy(); b2[2000, 3050] += 1
R["h2"] = hashlib.sha256(b2.tobytes()).hexdigest()
R["hex_diff"] = sum(a != b for a, b in zip(R["h1"], R["h2"]))
# content-addressed chunks: 121 tiles, edit the site window (4 tiles)
def chunks(a):
    return [hashlib.sha256(np.ascontiguousarray(a[i:i+512, j:j+512]).tobytes()).hexdigest()
            for i in range(0, 5490, 512) for j in range(0, 5490, 512)]
c1 = chunks(band); b3 = band.copy(); b3[2000:2100, 3050:3150] += 1; c3 = chunks(b3)
R["chunks"] = len(c1); R["chunks_changed"] = sum(a != b for a, b in zip(c1, c3))
R["changed_idx"] = [i for i, (a, b) in enumerate(zip(c1, c3)) if a != b]
R["snap1"] = hashlib.sha256("".join(c1).encode()).hexdigest()[:12]; R["snap2"] = hashlib.sha256("".join(c3).encode()).hexdigest()[:12]
# Icechunk
def tree_stats(d):
    out = {}
    for p in pathlib.Path(d).rglob("*"):
        if p.is_file():
            k = p.relative_to(d).parts[0]; n, s = out.get(k, (0, 0)); out[k] = (n + 1, s + p.stat().st_size)
    return out
d = tempfile.mkdtemp()
repo = icechunk.Repository.create(icechunk.local_filesystem_storage(d))
s = repo.writable_session("main")
arr = zarr.create_array(s.store, name="B12", shape=band.shape, chunks=(512, 512), dtype="uint16")
arr[:] = band; id1 = s.commit("ingest 2026-06-12")
R["ice_after1"] = tree_stats(d)
s = repo.writable_session("main"); arr = zarr.open_array(s.store, path="B12")
arr[2000:2100, 3050:3150] = b3[2000:2100, 3050:3150]; id2 = s.commit("reprocess site window")
R["ice_after2"] = tree_stats(d)
repo.create_tag("v1", snapshot_id=id1)
R["id1"], R["id2"] = id1, id2
old = zarr.open_array(repo.readonly_session(tag="v1").store, path="B12")[2000, 3050]
new = zarr.open_array(repo.readonly_session("main").store, path="B12")[2000, 3050]
R["old_px"], R["new_px"] = int(old), int(new)
# abort: uncommitted session leaves main unchanged
s = repo.writable_session("main"); zarr.open_array(s.store, path="B12")[0, 0] = 9; s.discard_changes()
R["main_after_discard"] = repo.lookup_branch("main") == id2
# conflict: two writers
sa = repo.writable_session("main"); sb = repo.writable_session("main")
zarr.open_array(sa.store, path="B12")[0, 0] = 1; zarr.open_array(sb.store, path="B12")[0, 0] = 2
sa.commit("writer A")
try: sb.commit("writer B"); R["conflict"] = "none"
except Exception as e: R["conflict"] = type(e).__name__
# same content, second repo: are ids content hashes?
d2 = tempfile.mkdtemp(); r2 = icechunk.Repository.create(icechunk.local_filesystem_storage(d2))
s2 = r2.writable_session("main"); a2 = zarr.create_array(s2.store, name="B12", shape=band.shape, chunks=(512, 512), dtype="uint16"); a2[:] = band
R["same_content_same_id"] = s2.commit("ingest 2026-06-12") == id1
R["history"] = [(a.id, a.message) for a in repo.ancestry(branch="main")]
print(json.dumps(R, indent=1, default=str))
(OUT / "results.json").write_text(json.dumps(R, indent=1, default=str) + "\n")


# %% [markdown]
# ## DVC and Icechunk together
#
# A two-stage DVC pipeline: `ingest` commits the band to Icechunk and writes the
# snapshot id to a small tracked file; `stats` reads exactly that snapshot. The
# pipeline is run, changed and run again, and an old git tag is checked out.

# %%
import subprocess
import sys

INGEST = r'''"""Stage 1: write the band into Icechunk and record the snapshot id."""

from __future__ import annotations

import json
import pathlib
import sys

import icechunk
import numpy as np
import yaml
import zarr

params = yaml.safe_load(pathlib.Path("params.yaml").read_text())
store_path = sys.argv[1]

# --- the band, and the site window if this version reprocesses it ---
rng = np.random.default_rng(0)
band = rng.integers(0, 4000, (5490, 5490), dtype=np.uint16)        # → (5490, 5490)
if params["reprocess_site"]:
    band[2000:2100, 3050:3150] += 1                                # touches 4 chunks

# --- one transaction on the Icechunk repository ---
storage = icechunk.local_filesystem_storage(store_path)
repo = icechunk.Repository.open_or_create(storage)
session = repo.writable_session("main")
if "B12" in zarr.open_group(session.store, mode="a"):
    array = zarr.open_array(session.store, path="B12")
    array[2000:2100, 3050:3150] = band[2000:2100, 3050:3150]
else:
    array = zarr.create_array(
        session.store, name="B12", shape=band.shape, chunks=(512, 512), dtype="uint16"
    )
    array[:] = band
snapshot_id = session.commit(f"B12, reprocess_site={params['reprocess_site']}")
# a tag keeps this snapshot from being expired while git still points at it
repo.create_tag(f"dvc-{snapshot_id}", snapshot_id=snapshot_id)

# --- the pointer: the only thing DVC and git see of the array ---
pointer = {"repo": store_path, "snapshot_id": snapshot_id}
pathlib.Path("snapshot.json").write_text(json.dumps(pointer, indent=2) + "\n")
'''

STATS = r'''"""Stage 2: read the pinned snapshot and write a metric."""

from __future__ import annotations

import json
import pathlib

import icechunk
import zarr

pointer = json.loads(pathlib.Path("snapshot.json").read_text())
storage = icechunk.local_filesystem_storage(pointer["repo"])
repo = icechunk.Repository.open(storage)
# read exactly the snapshot the pointer names, never "latest"
session = repo.readonly_session(snapshot_id=pointer["snapshot_id"])
array = zarr.open_array(session.store, path="B12")
window = array[2000:2100, 3050:3150]                               # → (100, 100)
metrics = {"site_mean": float(window.mean()), "site_pixel": int(window[0, 0])}
pathlib.Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
'''

DVC_YAML = """stages:
  ingest:
    cmd: {python} ingest.py {store}
    deps:
      - ingest.py
    params:
      - reprocess_site
    outs:
      - snapshot.json:
          cache: false
  stats:
    cmd: {python} stats.py
    deps:
      - stats.py
      - snapshot.json
    metrics:
      - metrics.json:
          cache: false
"""

work = pathlib.Path(tempfile.mkdtemp())
project, remote, store = work / "project", work / "remote", work / "store"
for folder in (project, remote, store):
    folder.mkdir()


def run(*command: str) -> str:
    done = subprocess.run(command, cwd=project, capture_output=True, text=True, env=dict(
        _os.environ, GIT_AUTHOR_NAME="demo", GIT_AUTHOR_EMAIL="demo@example.com",
        GIT_COMMITTER_NAME="demo", GIT_COMMITTER_EMAIL="demo@example.com"))
    if done.returncode != 0:
        raise RuntimeError(done.stdout + done.stderr)
    return done.stdout + done.stderr


(project / "ingest.py").write_text(INGEST)
(project / "stats.py").write_text(STATS)
(project / "dvc.yaml").write_text(DVC_YAML.format(python=sys.executable, store=store / "b12.icechunk"))
(project / "params.yaml").write_text("reprocess_site: false\n")
labels = ["scene,plume"] + [f"S2_{i:05d},{int(v < 0.1)}" for i, v in enumerate(np.random.default_rng(1).random(2000))]
(project / "labels.csv").write_text("\n".join(labels) + "\n")

run("git", "init", "-q", "-b", "main", ".")
run("dvc", "init", "-q")
run("dvc", "remote", "add", "-q", "-d", "drive", str(remote))   # a local folder stands in for Drive
run("dvc", "add", "-q", "labels.csv")
run("dvc", "repro")
run("git", "add", "-A"); run("git", "commit", "-q", "-m", "v1: ingest"); run("git", "tag", "v1")
again = run("dvc", "repro")
run("dvc", "push")
v1 = {**json.loads((project / "snapshot.json").read_text()), **json.loads((project / "metrics.json").read_text())}

(project / "params.yaml").write_text("reprocess_site: true\n")
run("dvc", "repro")
run("git", "add", "-A"); run("git", "commit", "-q", "-m", "v2: reprocess site"); run("git", "tag", "v2")
v2 = {**json.loads((project / "snapshot.json").read_text()), **json.loads((project / "metrics.json").read_text())}

run("git", "checkout", "-q", "v1")
run(sys.executable, "stats.py")
back = {**json.loads((project / "snapshot.json").read_text()), **json.loads((project / "metrics.json").read_text())}
run("git", "checkout", "-q", "main")


def tree(folder: pathlib.Path) -> tuple[int, int]:
    files = [f for f in folder.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


D = {
    "v1": {"snapshot_id": v1["snapshot_id"], "site_pixel": v1["site_pixel"], "site_mean": round(v1["site_mean"], 4)},
    "v2": {"snapshot_id": v2["snapshot_id"], "site_pixel": v2["site_pixel"], "site_mean": round(v2["site_mean"], 4)},
    "back_at_v1": {"snapshot_id": back["snapshot_id"], "site_pixel": back["site_pixel"]},
    "second_repro_skipped": "didn't change, skipping" in again,
    "dvc_remote_files": tree(remote)[0], "dvc_remote_kb": round(tree(remote)[1] / 1e3),
    "ice_files": tree(store)[0], "ice_mb": round(tree(store)[1] / 1e6),
    "dvc_version": run("dvc", "--version").strip(),
}
print(json.dumps(D, indent=1))
(OUT / "dvc_icechunk.json").write_text(json.dumps(D, indent=1) + "\n")
