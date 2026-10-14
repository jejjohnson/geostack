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
# # Report 12 · pipekit-experiment, run
#
# Regenerates every output in `docs/reports/12_pipekit_experiment.md`: the
# local and fsspec registries (stages 1 to 4 and 6), two trackers on one
# training loop and the artifact it returns (stage 5), and the Hydra, DVC and
# Metaflow adapters (stage 6). Registries, the DVC repository and the Metaflow
# flows are built in a temporary directory; each part ends in a cell asserting
# the outputs the page shows.
#
# Pinned revision: pipekit 0.0.2 at `ef3c17f` (pipekit-experiment, -train,
# -jax). It needs JAX, Equinox, Optax, fsspec, hydra-core, DVC and Metaflow;
# DVC and Metaflow run as subprocesses, so their CLIs must be on `PATH`.

# %%
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pipekit
from pipekit_experiment import (
    ExperimentTracker,
    InferenceArtifact,
    LocalModelRegistry,
    ModelRegistry,
    Run,
    RunMetrics,
    S3ModelRegistry,
    TrainingArtifact,
)
from pipekit_jax import JaxModelOp
from pipekit_train import Checkpoint, IterableDataset, LogToExperiment, MSE
from pipekit_train import TrainingLoop

root = Path(tempfile.mkdtemp(prefix="report12_"))
# Scale lives in its own module, so its registry state names an importable
# class (`r12ops.Scale`) for from_state, Hydra and the Metaflow subprocess.
(root / "r12ops.py").write_text(
    "from pipekit import Operator\n\n\n"
    "class Scale(Operator):\n"
    "    def __init__(self, factor: float = 1.0, offset: float = 0.0) -> None:\n"
    "        self.factor = factor\n"
    "        self.offset = offset\n\n"
    "    def _apply(self, x):\n"
    "        return x * self.factor + self.offset\n"
)
sys.path.insert(0, str(root))
from r12ops import Scale  # noqa: E402


def leaf_diff(a, b) -> float:
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


# %% [markdown]
# ## Part 1 · Content addressing and weights (stages 1 and 2)

# %%
reg = LocalModelRegistry(root / "reg")
is_registry = isinstance(reg, ModelRegistry)
h1 = reg.store(Scale(2.0, 1.0), name="s1", tags={"family": "scale", "v": 1})
same_twice = reg.store(Scale(2.0, 1.0)) == h1
config_change = reg.store(Scale(2.0, 1.5)) != h1
h3 = reg.store(Scale(2.0, 1.0), weights=b"abc")
h3b = reg.store(Scale(2.0, 1.0), weights=b"abd")
weight_change = h3 != h1 and h3 != h3b
scale_loaded = reg.load("s1").state

op = JaxModelOp(eqx.nn.Linear(40, 40, key=jax.random.key(0)))
hj = reg.store(op, name="lin40", weights=op.serialize_weights(),
               tags={"family": "jax"})
other = JaxModelOp(eqx.nn.Linear(40, 40, key=jax.random.key(99)))
weights_distinguish = reg.store(other, weights=other.serialize_weights()) != hj
stored = {p.name: p.stat().st_size for p in (reg.root / hj).iterdir()}
operator_json = json.loads((reg.root / hj / "operator.json").read_text())
try:
    reg.load(hj)
    jax_load = "ok"
except RuntimeError as e:
    jax_load = f"RuntimeError: {e}"
skel = JaxModelOp(eqx.nn.Linear(40, 40, key=jax.random.key(7)))
skeleton_diff = max(leaf_diff(skel.from_registry(reg, ref).module, op.module)
                    for ref in (hj, "lin40"))
try:
    JaxModelOp(eqx.nn.Linear(40, 20, key=jax.random.key(0))).from_registry(
        reg, "lin40")
    mismatch = ""
except RuntimeError as e:
    mismatch = str(e)

L = LocalModelRegistry(root / "weightless")
collide = (L.store(JaxModelOp(eqx.nn.Linear(40, 40, key=jax.random.key(0))))
           == L.store(JaxModelOp(eqx.nn.Linear(3, 7, key=jax.random.key(1)))))
stored, operator_json, collide

# %%
assert is_registry and same_twice and config_change and weight_change
assert scale_loaded["config"] == {"factor": 2.0, "offset": 1.0}
assert weights_distinguish and len(hj) == 64
assert stored["weights.bin"] == 6816 and set(stored) == {
    "metadata.json", "operator.json", "weights.bin"}
assert operator_json == {"class": "JaxModelOp", "config": {"module_class": "Linear"},
                         "module": "pipekit_jax.model_op"}
assert (reg.root / "_tags" / "lin40").read_text().strip() == hj
assert jax_load.startswith("RuntimeError: from_state cannot reconstruct JaxModelOp")
assert "forbid_in_yaml" in jax_load
assert skeleton_diff == 0.0 and "changed shape" in mismatch
assert collide  # weightless JaxModelOps of different shapes share a hash

# %% [markdown]
# ## Part 2 · Tags and promotion (stage 3)

# %%
reg.tag(h3, "production")
try:
    reg.tag(h3b, "production")
    retag = ""
except FileExistsError as e:
    retag = str(e)
reg.tag(h3b, "production", force=True)
after_force = reg.resolve_tag("production")
old_mentioned = [p.name for p in (reg.root / "_tags").iterdir()
                 if h3 in p.read_text()]
reg.store(Scale(9.0, 9.0), name="production")  # moves the tag, silently
silent_move = reg.resolve_tag("production") != h3b
by_family = reg.list(tags={"family": "scale"})
str_vs_int = reg.list(tags={"v": "1"})
missing = [m for m in ("gc", "untag", "delete", "history") if not hasattr(reg, m)]
retag, old_mentioned, silent_move, str_vs_int, missing

# %%
assert retag == "Tag 'production' already exists. Pass force=True to overwrite."
assert after_force == h3b and old_mentioned == [] and silent_move
assert by_family == [h1] and str_vs_int == []
assert missing == ["gc", "untag", "delete", "history"]

# %% [markdown]
# ## Part 3 · Two trackers on one loop, and the artifacts (stages 4 and 5)

# %%
class MemTracker:
    """An ExperimentTracker that keeps runs in memory and returns a Run."""

    def __init__(self) -> None:
        self.series: dict[str, RunMetrics] = {}
        self.calls: list[tuple] = []

    def start_run(self, name, config):
        r = Run(run_id=uuid.uuid4().hex, pipeline_hash="", config=dict(config))
        self.series[r.run_id] = RunMetrics()
        self.calls.append(("start", name))
        return r

    def log_metrics(self, run, metrics, step):
        for k, v in metrics.items():
            self.series[run.run_id].log(k, v, step)

    def log_artifact(self, run, path, name):
        run.artifacts[name] = path

    def end_run(self, run, status):
        run.status = status
        run.ended_at = datetime.now(UTC)
        self.calls.append(("end", status))


class Partial:
    def start_run(self, name, config): ...


is_tracker = (isinstance(MemTracker(), ExperimentTracker),
              isinstance(Partial(), ExperimentTracker))

rng = np.random.default_rng(0)
xs = rng.normal(size=(200, 3)).astype(np.float32)
ys = (xs @ np.array([[1.0], [-2.0], [0.5]], np.float32) + 0.3).astype(np.float32)
ds = IterableDataset(source=list(zip(xs, ys, strict=True)),
                     content_hash="r12-linear-200-0")
ta, tb = MemTracker(), MemTracker()
reg5 = LocalModelRegistry(root / "reg5")
la, lb = LogToExperiment(ta, run_name="a"), LogToExperiment(tb, run_name="b")
loop = TrainingLoop(model_op=JaxModelOp(eqx.nn.Linear(3, 1, key=jax.random.key(0))),
                    dataset=ds, loss=MSE(),
                    optimizer_config={"name": "adam", "lr": 5e-2}, max_steps=50,
                    batch_size=16, backend="equinox", seed=0,
                    callbacks=(la, lb, Checkpoint(registry=reg5)))
trained, art = loop.run()
sa, sb = ta.series[la.run.run_id].values, tb.series[lb.run.run_id].values
key = next(iter(sa))
stream = [v for _, v in sa[key]]

art_path = root / "train_art.json"
art.save(art_path)
art_round_trip = TrainingArtifact.load(art_path) == art
try:
    art.reload_model(reg5)
    reload_model = "ok"
except RuntimeError as e:
    reload_model = f"RuntimeError: {e}"
try:
    pipekit.loads(art.training_pipeline_yaml)
    loop_loads = "ok"
except RuntimeError as e:
    loop_loads = f"RuntimeError: {e}"

# stage 4: an InferenceArtifact pins by name, so the pin drifts with the tag
hs = reg5.store(Scale(3.0, 0.0), name="production")
inf = InferenceArtifact(pipeline_yaml="", pinned_model_hashes={"steps.1": "production"})
inf.save(root / "inf.json")
inf_round_trip = InferenceArtifact.load(root / "inf.json") == inf
pinned = {k: v.state["config"] for k, v in inf.reload_models(reg5).items()}
(type(art).__name__, len(stream), stream[0], stream[-1], art.tracker_run_id,
 art.model_registry_uri, pinned)

# %%
assert is_tracker == (True, False)
assert sa == sb and len(stream) == 50
assert (round(stream[0], 2), round(stream[-1], 4)) == (3.28, 0.0029)
assert la.run.status == lb.run.status == "completed"
assert type(art) is TrainingArtifact and art_round_trip
assert art.tracker_run_id is None  # Run has no __getitem__ (loop.py:416)
assert art.model_registry_uri == f"registry://{art.trained_model_hash}"
assert art.model_registry_uri.startswith("registry://d120fe9b")
assert reg5.load_weights(art.trained_model_hash) is None
assert reload_model.startswith("RuntimeError: from_state cannot reconstruct "
                               "EquinoxModelOp")
assert loop_loads.startswith("RuntimeError: from_state cannot reconstruct "
                             "TrainingLoop")
assert inf_round_trip and pinned == {"steps.1": {"factor": 3.0, "offset": 0.0}}
assert not hasattr(TrainingArtifact, "rerun")
assert not hasattr(reg5, "location")  # LocalModelRegistry has root, not uri
assert hasattr(reg5, "root") and not hasattr(reg5, "uri")

# %% [markdown]
# ## Part 4 · The adapters and the fsspec registry (stage 6)

# %%
from pipekit_experiment.adapters.dvc import DVCDatasetVersioning  # noqa: E402
from pipekit_experiment.adapters.hydra import HydraConfigLoader as HL  # noqa: E402

flat = HL.from_yaml(HL.to_yaml(Scale(2.0, 1.0)))(3.0)
seq = pipekit.Sequential([Scale(2.0, 1.0), Scale(0.5, 0.0)])
nested_yaml = HL.to_yaml(seq)
try:
    HL.from_yaml(nested_yaml)
    nested = "ok"
except Exception as e:  # hydra raises InstantiationException
    nested = f"{type(e).__name__}: {e}"
hand_written = HL.from_hydra_cfg({
    "_target_": "pipekit.Sequential",
    "operators": [{"_target_": "r12ops.Scale", "factor": 2.0, "offset": 1.0},
                  {"_target_": "r12ops.Scale", "factor": 0.5}]})(3.0)

repo = root / "dvcrepo"
(repo / "data").mkdir(parents=True)
(repo / "data" / "train.csv").write_text("a,b\n1,2\n")
for cmd in (["git", "init", "-q"], ["dvc", "init", "-q"]):
    subprocess.run(cmd, cwd=repo, check=True)
dvc = DVCDatasetVersioning(repo=str(repo))
dvc.add("data/train.csv")
dvc_hash = dvc.hash("data/train.csv")
try:
    dvc.push("data/train.csv")
    dvc_push = "ok"
except subprocess.CalledProcessError as e:
    dvc_push = f"CalledProcessError: {e.stderr or ''}"

flows = {
    "adapter": """
from metaflow import FlowSpec, step
from pipekit_experiment.adapters.metaflow import MetaflowStepAdapter
from r12ops import Scale
class AdapterFlow(FlowSpec):
    @step
    def start(self):
        self.raw = 3.0
        self.next(self.scaled)
    scaled = MetaflowStepAdapter.as_step(Scale(2.0, 1.0), name="scaled",
                                         inputs=["raw"])
    @step
    def end(self):
        print("RESULT", self.scaled)
if __name__ == "__main__":
    AdapterFlow()
""",
    "manual": """
from metaflow import FlowSpec, step
from pipekit_experiment.adapters.metaflow import MetaflowStepAdapter
from r12ops import Scale
import pipekit
class ManualFlow(FlowSpec):
    @step
    def start(self):
        self.raw = 3.0
        self.next(self.scale)
    @step
    def scale(self):
        seq = pipekit.Sequential([Scale(2.0, 1.0), Scale(0.5, 0.0)])
        MetaflowStepAdapter.run(seq, self, name="scaled", inputs=["raw"])
        self.next(self.end)
    @step
    def end(self):
        print("RESULT", self.scaled)
if __name__ == "__main__":
    ManualFlow()
""",
}
mf = {}
env = os.environ | {"PYTHONPATH": os.pathsep.join([str(root), *sys.path]),
                    "USERNAME": "report12", "METAFLOW_DEFAULT_DATASTORE": "local"}
for name, src in flows.items():
    (root / f"flow_{name}.py").write_text(src)
    p = subprocess.run([sys.executable, f"flow_{name}.py", "--no-pylint", "run"],
                       cwd=root, env=env, capture_output=True, text=True)
    mf[name] = (p.returncode, p.stdout + p.stderr)

# the fsspec registry: same hashes and layout as the local one
mem = S3ModelRegistry("memory://r12/models")
try:
    mem.list()
    empty_list = "ok"
except FileNotFoundError:
    empty_list = "FileNotFoundError"
hm = mem.store(Scale(2.0, 1.0), name="s1", tags={"k": "v"}, weights=b"w")
same_as_local = hm == LocalModelRegistry(root / "cmp").store(Scale(2.0, 1.0),
                                                              weights=b"w")
fdir = root / "fsfile"
fdir.mkdir()
try:
    S3ModelRegistry(fdir.as_uri()).store(Scale(2.0, 1.0), weights=b"w")
    file_store = "ok"
except FileNotFoundError:
    file_store = "FileNotFoundError"
auto = S3ModelRegistry(fdir.as_uri(), storage_options={"auto_mkdir": True})
auto_ok = auto.load_weights(auto.store(Scale(2.0, 1.0), weights=b"w")) == b"w"
flat, nested, hand_written, dvc_hash, empty_list, file_store, auto_ok

# %%
assert flat == 7.0 and hand_written == 3.5
assert "class: Scale" in nested_yaml and "_target_: r12ops" not in nested_yaml
assert nested.startswith("InstantiationException")
assert "Sequential.operators[0] must be an Operator, got DictConfig" in nested
assert dvc_hash == "e5ebd4c02cefbe7955977c67ada242b7"
assert dvc_push.startswith("CalledProcessError") and "no remote" in dvc_push
assert mf["adapter"][0] != 0
assert "missing a self.next() transition" in mf["adapter"][1]
assert mf["manual"][0] == 0 and "RESULT 3.5" in mf["manual"][1]
assert empty_list == "FileNotFoundError" and same_as_local
assert file_store == "FileNotFoundError" and auto_ok


# %% [markdown]
# ## The readiness figure
#
# The audit's readiness figure counts the markers on this page's own sketches
# (`today`, `proposed (…)`, `user`); `docs/images/reports/make_figures.py`
# draws it. These are the counts its caption quotes.

# %%
import importlib.util
import sys
from collections import Counter
from pathlib import Path

root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
spec = importlib.util.spec_from_file_location(
    "report_figures", root / "docs/images/reports/make_figures.py")
rf = importlib.util.module_from_spec(spec)
sys.modules["report_figures"] = rf
spec.loader.exec_module(rf)
calls = rf.marked_calls(next((root / "docs/reports").glob("12_*.md")))
ready = Counter(c.status for c in calls)
ready

# %%
assert len(calls) == 21
assert ready == Counter({'here': 3, 'today': 17, 'user': 1})
