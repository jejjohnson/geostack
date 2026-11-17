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
# # Background · backends and APIs: exercise the service and time the blocking trap
#
# Regenerates every measured number on
# `docs/design/pages/backends-apis.html` (stub: `docs/design/backends_apis.md`).
# Starts `backends_service.py` under uvicorn, then calls every endpoint: a job from
# request to product, the same request repeated, twelve identical requests at once,
# a malformed request, a missing product, and twenty concurrent reads against each of
# the three read variants.
#
# Needs: everything `geo_pipeline.py` needs, plus FastAPI, uvicorn and httpx.
# It illustrates a docs page and is not a stage of the DVC pipeline; run it by hand:
#
#     python notebooks/background/backends.py
#
# It writes `docs/images/background_backends/results.json`, the file the page's
# numbers were taken from. Timings depend on the machine and change from run to run; counts, sizes and errors do not. About half a minute. Uses local port 8765.

# %%
"""Start the service, exercise every endpoint, and time the blocking trap."""
import asyncio, json, pathlib, shutil, subprocess, sys, time
import httpx

HERE = pathlib.Path(__file__).resolve().parent
_root = HERE
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
import importlib.util, os, tempfile
OUT = pathlib.Path(os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_backends"))
OUT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path(tempfile.mkdtemp())
# the archive is the one the geo-pipeline notebook writes
_spec = importlib.util.spec_from_file_location("geo_pipeline", HERE / "geo_pipeline.py")
_pipeline = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pipeline)
_pipeline.ARCHIVE = WORK / "archive"; _pipeline.make_archive()
BASE = "http://127.0.0.1:8765"
proc = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "uvicorn", "backends_service:app", "--port", "8765", "--log-level", "warning"],
                        cwd=HERE, stdout=subprocess.DEVNULL, stderr=open(WORK / "server.log", "w"),
                        env=dict(os.environ, BACKGROUND_ARCHIVE=str(WORK / "archive"), BACKGROUND_STATE=str(WORK / "state")))
R = {}
try:
    with httpx.Client(base_url=BASE, timeout=60) as c:
        for _ in range(200):
            try:
                if c.get("/health").status_code == 200: break
            except httpx.TransportError: time.sleep(0.1)
        R["empty"] = c.get("/products").json()
        body = {"bounds": [723000.0, 4375000.0, 728000.0, 4379000.0], "start": "2026-06-01", "end": "2026-06-30", "recipe": "ndvi"}
        t = time.perf_counter(); r = c.post("/jobs", json=body); R["post_ms"] = round((time.perf_counter() - t) * 1e3, 1)
        R["post_status"], R["job"] = r.status_code, r.json()
        r2 = c.post("/jobs", json=body); R["repost_status"], R["repost_same"] = r2.status_code, r2.json()["id"] == r.json()["id"]
        polls = 0
        while True:
            j = c.get(f"/jobs/{r.json()['id']}").json(); polls += 1
            if j["status"] in ("done", "failed"): break
            time.sleep(0.05)
        R["job_ms"] = round((time.perf_counter() - t) * 1e3, 1); R["polls"] = polls; R["job_done"] = j
        # twelve identical requests at the same instant: exactly one may create the job
        import concurrent.futures
        other = dict(body, start="2026-06-05", end="2026-06-25")
        with concurrent.futures.ThreadPoolExecutor(12) as pool:
            burst = list(pool.map(lambda _: httpx.post(BASE + "/jobs", json=other, timeout=60), range(12)))
        R["burst"] = {"sent": 12, "created_202": sum(x.status_code == 202 for x in burst),
                      "existing_200": sum(x.status_code == 200 for x in burst),
                      "errors": sum(x.status_code >= 400 for x in burst), "job_ids": len({x.json()["id"] for x in burst})}
        while c.get(f"/jobs/{burst[0].json()['id']}").json()["status"] not in ("done", "failed"): time.sleep(0.05)
        R["burst"]["products_for_it"] = sum(p["id"] == f"ndvi_{burst[0].json()['id']}" for p in c.get("/products").json())
        r3 = c.post("/jobs", json=body); R["repost_after_status"], R["repost_after"] = r3.status_code, r3.json()
        R["search"] = c.get("/products", params={"bbox": "724000,4376000,725000,4377000", "start": "2026-06-01", "end": "2026-06-02"}).json()
        R["search_miss"] = c.get("/products", params={"bbox": "100000,4376000,101000,4377000"}).json()
        R["get"] = c.get(f"/products/{j['product_id']}").json()
        r404 = c.get("/products/nope"); R["404"] = [r404.status_code, r404.json()]
        bad = dict(body, bounds=[1.0, 2.0, 3.0]); r422 = c.post("/jobs", json=bad); R["422"] = [r422.status_code, r422.json()["detail"][0]["msg"], r422.json()["detail"][0]["loc"]]
        spec = c.get("/openapi.json").json(); R["paths"] = sorted(spec["paths"]); R["schemas"] = sorted(spec["components"]["schemas"])
        t = time.perf_counter(); c.get("/products"); R["search_ms"] = round((time.perf_counter() - t) * 1e3, 2)

    async def trap(path, n=20):
        async with httpx.AsyncClient(base_url=BASE, timeout=60) as ac:
            async def probe():
                await asyncio.sleep(0.05)
                t0 = time.perf_counter(); await ac.get("/health"); return time.perf_counter() - t0
            t0 = time.perf_counter()
            out = await asyncio.gather(probe(), *(ac.get(path) for _ in range(n)))
            return round(time.perf_counter() - t0, 2), round(out[0] * 1e3, 1)
    for name in ("blocking", "plain", "offloaded"):
        best = None
        for _ in range(2):
            wall, health = asyncio.run(trap(f"/read/{name}"))
            if best is None or wall < best[0]: best = (wall, health)
        R[f"trap_{name}"] = {"wall_s": best[0], "health_ms": best[1]}
finally:
    proc.terminate(); proc.wait()
R["lines"] = sum(1 for _ in open(HERE / "backends_service.py"))
print(json.dumps(R, indent=1))
(OUT / "results.json").write_text(json.dumps(R, indent=1) + "\n")
