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
# # Background · databases and catalogs: one question, five ways to answer it
#
# Regenerates every measured number on
# `docs/design/pages/databases-catalogs.html` (stub: `docs/design/databases_catalogs.md`).
# Builds a synthetic catalog of 100 000 scenes and times the question “which scenes
# cover the site in June 2026” as a scan, with a B-tree, with an R-tree and over Parquet;
# then an aggregate in a row store and a column store, an upsert and a rollback.
#
# Needs: NumPy, DuckDB, pyarrow (SQLite ships with Python, with its R-tree module).
# The third-party packages are in the `background` pixi environment. Not a stage of the DVC pipeline; run it by hand:
#
#     pixi run -e background python notebooks/background/databases.py
#
# It writes `docs/images/background_databases/results.json`, the file the page's
# numbers were taken from. Timings depend on the machine and change from run to run; counts, sizes and errors do not. Under a minute.

# %%
"""Catalog of 100,000 scenes: find the ones over a site in June 2026."""
import json, os, sqlite3, time, datetime as dt
import numpy as np, duckdb, pyarrow as pa, pyarrow.parquet as pq

# %%
# Where the recorded output goes. Set BACKGROUND_OUT to write somewhere else.
import os as _os
import pathlib as _pathlib

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(_os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_databases"))
OUT.mkdir(parents=True, exist_ok=True)

import tempfile

_os.chdir(tempfile.mkdtemp())  # scratch files: cat.sqlite, cat.parquet

N = 100_000
rng = np.random.default_rng(0)
# scenes over Iberia 2023-01-01 .. 2026-12-31, 110 km footprints
lon = rng.uniform(-10, 4, N); lat = rng.uniform(36, 44, N)
dx = 1.3; dy = 1.0
t0 = dt.datetime(2023, 1, 1); span = (dt.datetime(2027, 1, 1) - t0).total_seconds()
ts = np.sort(rng.uniform(0, span, N))
when = [(t0 + dt.timedelta(seconds=float(s))).isoformat() for s in ts]
cloud = rng.uniform(0, 100, N).round(1)
ids = [f"S2_{i:06d}" for i in range(N)]
href = [f"s3://sentinel-cogs/scene/{i:06d}/B12.tif" for i in range(N)]
rows = list(zip(ids, when, (lon - dx/2).tolist(), (lon + dx/2).tolist(),
                (lat - dy/2).tolist(), (lat + dy/2).tolist(), cloud.tolist(), href))
SITE = (-0.40, 39.45)          # lon, lat
A, B = "2026-06-01", "2026-07-01"

def best(f, reps=5):
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return round(b * 1e3, 3), r

R = {"n": N}
def scan():
    x, y = SITE
    return sum(1 for r in rows if A <= r[1] < B and r[2] <= x <= r[3] and r[4] <= y <= r[5])
R["py_scan_ms"], R["hits"] = best(scan)

db = sqlite3.connect("cat.sqlite"); db.executescript(
    "drop table if exists scenes; drop table if exists scenes_rt;"
    "create table scenes(id text primary key, datetime text, x0 real, x1 real, y0 real, y1 real, cloud real, href text);")
db.executemany("insert into scenes values (?,?,?,?,?,?,?,?)", rows); db.commit()
q = "select count(*) from scenes where datetime >= ? and datetime < ? and x0 <= ? and x1 >= ? and y0 <= ? and y1 >= ?"
args = (A, B, SITE[0], SITE[0], SITE[1], SITE[1])
R["sqlite_scan_ms"], h = best(lambda: db.execute(q, args).fetchone()[0]); assert h == R["hits"]
db.execute("create index ix_time on scenes(datetime)"); db.commit()
R["sqlite_btree_ms"], h = best(lambda: db.execute(q, args).fetchone()[0]); assert h == R["hits"]
R["june_rows"] = db.execute("select count(*) from scenes where datetime >= ? and datetime < ?", (A, B)).fetchone()[0]
db.execute("create virtual table scenes_rt using rtree(rid, x0, x1, y0, y1)")
db.execute("insert into scenes_rt select rowid, x0, x1, y0, y1 from scenes"); db.commit()
q2 = ("select count(*) from scenes_rt r join scenes s on s.rowid = r.rid "
      "where r.x0 <= ? and r.x1 >= ? and r.y0 <= ? and r.y1 >= ? and s.datetime >= ? and s.datetime < ?")
R["sqlite_rtree_ms"], h = best(lambda: db.execute(q2, (SITE[0], SITE[0], SITE[1], SITE[1], A, B)).fetchone()[0]); assert h == R["hits"]
R["site_rows"] = db.execute("select count(*) from scenes_rt where x0 <= ? and x1 >= ? and y0 <= ? and y1 >= ?", (SITE[0], SITE[0], SITE[1], SITE[1])).fetchone()[0]
# point lookup by id
R["sqlite_get_ms"], _ = best(lambda: db.execute("select href from scenes where id = ?", ("S2_077777",)).fetchone())
# analytics: mean cloud by month — row store vs column store
qa = "select substr(datetime,1,7) m, avg(cloud) from scenes group by m"
R["sqlite_agg_ms"], _ = best(lambda: db.execute(qa).fetchall())
tbl = pa.table({"id": ids, "datetime": pa.array([dt.datetime.fromisoformat(w) for w in when]),
                "x0": lon - dx/2, "x1": lon + dx/2, "y0": lat - dy/2, "y1": lat + dy/2, "cloud": cloud, "href": href})
pq.write_table(tbl, "cat.parquet", row_group_size=10_000, compression="zstd")
R["parquet_mb"] = round(os.path.getsize("cat.parquet") / 1e6, 2); R["sqlite_mb"] = round(os.path.getsize("cat.sqlite") / 1e6, 2)
md = pq.ParquetFile("cat.parquet").metadata
R["row_groups"] = md.num_row_groups
con = duckdb.connect()
qd = (f"select count(*) from 'cat.parquet' where datetime >= TIMESTAMP '{A}' and datetime < TIMESTAMP '{B}' "
      f"and x0 <= {SITE[0]} and x1 >= {SITE[0]} and y0 <= {SITE[1]} and y1 >= {SITE[1]}")
R["duckdb_parquet_ms"], h = best(lambda: con.execute(qd).fetchone()[0]); assert h == R["hits"]
R["duckdb_agg_ms"], _ = best(lambda: con.execute("select date_trunc('month', datetime) m, avg(cloud) from 'cat.parquet' group by m").fetchall())
# row groups touched by the June filter
jun0, jun1 = dt.datetime(2026, 6, 1), dt.datetime(2026, 7, 1)
ci = md.schema.names.index("datetime") if hasattr(md.schema, "names") else 1
touched = 0
for g in range(md.num_row_groups):
    st = md.row_group(g).column(1).statistics
    if st.max >= jun0 and st.min < jun1: touched += 1
R["row_groups_touched"] = touched
# upsert + transaction rollback
db.execute("insert into scenes(id, datetime, x0,x1,y0,y1,cloud,href) values ('S2_000001','2026-06-12',0,1,0,1,5,'new') "
           "on conflict(id) do update set href = excluded.href, cloud = excluded.cloud"); db.commit()
R["upsert_count"] = db.execute("select count(*) from scenes").fetchone()[0]
try:
    with db:
        db.execute("update scenes set href = 'half' where id = 'S2_000002'")
        raise RuntimeError("crash between two writes")
except RuntimeError: pass
R["after_rollback"] = db.execute("select href from scenes where id = 'S2_000002'").fetchone()[0]
print(json.dumps(R, indent=1))
(OUT / "results.json").write_text(json.dumps(R, indent=1) + "\n")
