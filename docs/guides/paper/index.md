# Guides from the paper

These two guides accompany the SpatialData-DB paper. They were run on a GPU node against the dated catalog release 2026-10-28, and the pages show their stored outputs; the docs build does not run them again. Each guide pins `sddb.Catalog(version="2026-10-28")`, so a new run reads the same catalog rows.

- {doc}`he-expression` predicts gene expression from H&E images with a frozen pathology encoder (H-optimus-0) and ridge regression, and evaluates on held-out study groups.
- {doc}`novae-domains` builds a mouse brain cohort from MERFISH, MERSCOPE and Xenium and runs the pretrained Novae model on it without training, to compare spatial domains across platforms.

Both need a GPU for a reasonable run time and packages beyond the client (`timm` and `torch` for the first, `novae` for the second). The H&E guide also needs access to H-optimus-0 on Hugging Face.

```{toctree}
:hidden:

he-expression
novae-domains
```
