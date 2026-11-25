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
# # Background · parallelism: 121 tiles, serial, async, threads, processes
#
# Regenerates every measured number on
# `docs/design/pages/parallelism.html` (stub: `docs/design/parallelism.md`).
# Reads 121 tiles with a 50 ms injected wait and a real deflate decode, under every
# mechanism the page compares, and times each (best of three).
#
# Needs: NumPy and OmegaConf, both in the default pixi environment. Every setting
# (seed, tile count and size, latency, worker counts) is read from
# `configs/background/parallelism.yaml` and recorded in the output.
# It illustrates a docs page and is not a stage of the DVC pipeline; run it by hand:
#
#     pixi run python notebooks/background/parallelism.py
#
# It writes `docs/images/background_parallelism/results.json`, the file the page's
# numbers were taken from. Timings depend on the machine and change from run to run; counts, sizes and errors do not. About one minute.

# %%
# Settings come from configs/background/parallelism.yaml; the recorded output goes to
# docs/images/background_parallelism (set BACKGROUND_OUT to write somewhere else).
import os as _os
import pathlib as _pathlib

from omegaconf import OmegaConf

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(_os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_parallelism"))
OUT.mkdir(parents=True, exist_ok=True)
CFG = OmegaConf.load(_root / "configs/background/parallelism.yaml")

# %%
"""Running example: 121 tiles; fetch = 50 ms wait, decode = deflate + predictor."""
import asyncio, json, os, time, zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

N, LAT, PX = CFG.tiles, CFG.latency_s, CFG.tile_px
rng = np.random.default_rng(CFG.seed)
base = np.cumsum(rng.integers(-40, 40, (PX, PX)), axis=1).astype(np.int16)
diff = np.diff(base, axis=1, prepend=0).astype(np.int16)
BLOB = zlib.compress(diff.tobytes(), CFG.deflate_level)

def fetch(i):
    time.sleep(LAT); return BLOB
def decode(blob):                      # native: zlib + numpy release the GIL
    a = np.frombuffer(zlib.decompress(blob), np.int16).reshape(PX, PX)
    return int(np.cumsum(a, axis=1, dtype=np.int16).sum())
def decode_py(blob):                   # pure Python: holds the GIL
    s = 0
    for b in blob[:60000]: s = (s * 31 + b) & 0xFFFF
    return s
def tile(i): return decode(fetch(i))
def cpu_native(i): return decode(BLOB)
def cpu_py(i): return decode_py(BLOB)

def timeit(f, reps=CFG.repeats):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return round(best, 3)

async def afetch(i, sem):
    async with sem:
        await asyncio.sleep(LAT); return BLOB
async def run_async(conc, offload):
    sem = asyncio.Semaphore(conc)
    async def one(i):
        blob = await afetch(i, sem)
        return await asyncio.to_thread(decode, blob) if offload else decode(blob)
    return await asyncio.gather(*(one(i) for i in range(N)))
async def run_async_blocking(conc):   # the trap: time.sleep inside a coroutine
    sem = asyncio.Semaphore(conc)
    async def one(i):
        async with sem:
            time.sleep(LAT); return decode(BLOB)
    return await asyncio.gather(*(one(i) for i in range(N)))

def pool(P, n, f):
    with P(n) as ex: list(ex.map(f, range(N)))

if __name__ == "__main__":
    R = {"config": OmegaConf.to_container(CFG), "cores": os.cpu_count(), "blob_kb": round(len(BLOB)/1024, 1)}
    R["decode_ms"] = round(timeit(lambda: [decode(BLOB) for _ in range(N)]) / N * 1e3, 2)
    R["decode_py_ms"] = round(timeit(lambda: [decode_py(BLOB) for _ in range(N)]) / N * 1e3, 2)
    R["serial"] = timeit(lambda: [tile(i) for i in range(N)], 1)
    for n in CFG.thread_workers:
        R[f"threads{n}"] = timeit(lambda: pool(ThreadPoolExecutor, n, tile))
    for c in CFG.async_in_flight:
        R[f"async{c}"] = timeit(lambda: asyncio.run(run_async(c, False)))
    c = CFG.async_offload_in_flight
    R[f"async{c}_offload"] = timeit(lambda: asyncio.run(run_async(c, True)))
    c = CFG.async_blocking_in_flight
    R[f"async{c}_blocking"] = timeit(lambda: asyncio.run(run_async_blocking(c)), 1)
    for n in CFG.process_workers:
        R[f"procs{n}"] = timeit(lambda: pool(ProcessPoolExecutor, n, tile), 2)
    # CPU only
    w = CFG.cpu_workers
    R["cpu_py_serial"] = timeit(lambda: [cpu_py(i) for i in range(N)])
    R[f"cpu_py_threads{w}"] = timeit(lambda: pool(ThreadPoolExecutor, w, cpu_py))
    R[f"cpu_py_procs{w}"] = timeit(lambda: pool(ProcessPoolExecutor, w, cpu_py))
    R["cpu_nat_serial"] = timeit(lambda: [cpu_native(i) for i in range(N)])
    R[f"cpu_nat_threads{w}"] = timeit(lambda: pool(ThreadPoolExecutor, w, cpu_native))
    R[f"cpu_nat_procs{w}"] = timeit(lambda: pool(ProcessPoolExecutor, w, cpu_native))
    # vectorise: reflectance = (DN - 1000) / 10000 on one tile
    row = base.tolist()
    def loop(): return [[(v - 1000) / 10000 for v in r] for r in row]
    def vec(): return (base.astype(np.float32) - 1000) / 10000
    R["scale_loop_ms"] = round(timeit(loop) * 1e3, 2)
    R["scale_numpy_ms"] = round(timeit(lambda: [vec() for _ in range(100)]) / 100 * 1e3, 3)
    # pickling cost: send a decoded tile back from a process
    def ret(i): return base
    import pickle
    R["pickle_tile_ms"] = round(timeit(lambda: [pickle.loads(pickle.dumps(base)) for _ in range(100)]) / 100 * 1e3, 3)
    R["proc_startup_ms"] = round(timeit(lambda: pool(ProcessPoolExecutor, 2, abs), 2) * 1e3, 1)
    print(json.dumps(R, indent=1))
    (OUT / "results.json").write_text(json.dumps(R, indent=1) + "\n")
