# Orchestration: above the ladder

**What it is.** Running recipes over many items, on a schedule, behind an API, with retries and storage. It is not a rung, because it runs a recipe without changing it.

**Examples.** A DVC stage, an Airflow or Prefect task, a FastAPI endpoint, a GeoApps endpoint, a sweep over one setting.

**How to tell.** It decides when, where and how often, never what is computed. Its inputs are a recipe and a list of items.

**How to explain it.** Say what the unit of work is (one scene, one day, one tile), what happens on failure, and where results go. The maths was finished at the recipe.

**How it goes wrong.** A setting that changes the result and lives only in the scheduler is a recipe setting in the wrong place.
