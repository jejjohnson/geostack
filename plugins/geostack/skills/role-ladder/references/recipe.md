# ⑥ Recipe: the paragraph

**What it is.** Steps in order, with their settings, making one job from source to product. In code it is a `Sequential` or a `Graph` of steps, itself an Operator; as data it is the YAML or JSON that loads into one.

**Kind.** An Operator.

**Examples.** Combinators: `Sequential`, `Graph`, `Branch`, `Retry`. Templates: `DACycle`, `TrainingLoop`, `SpatialPatcher`. As data: `pipekit_ops.dumps`, `pipekit_ops.loads`, a `recipe.yaml`. In the methane example, `plume_recipe`.

**How to tell.** It holds configs only and does no computing of its own. It can be summarised without data and saved.

**How to explain it.** Draw the steps as boxes with what flows between them, and say why the shape is what it is: the plume recipe is a graph, not a chain, because the enhancement feeds two steps and the wind enters on its own grid.

**How it goes wrong.** Logic written between the steps belongs in a step. A recipe that cannot be written as data usually has a setting hidden in code.
