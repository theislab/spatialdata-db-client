---
orphan: true
---

# Use cases

The use cases are longer analyses built on the steps of the {doc}`tutorials <../tutorials/index>`. Each starts
from a question, builds a cohort for it and runs the analysis to a result.

- {doc}`training_cohort` prepares a cohort for machine learning: one table across datasets, a split that keeps
  each experiment in one fold, and a manifest that records the split. It runs on a laptop in a few minutes.
- {doc}`visium_atlas/visium-atlas` integrates every Visium section in the catalog, about 4,000 of them, into one
  atlas and asks whether the spots group by species or by organ. It needs a large GPU node and has its own
  environment.

```{toctree}
:hidden:

training_cohort
visium_atlas/visium-atlas
```
