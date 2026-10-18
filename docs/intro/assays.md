# Spatial assays

{{ intro_count }}
Each dataset loads as a [SpatialData](https://spatialdata.scverse.org) object whose elements
(images, shapes, points, labels, tables) depend on the assay.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Visium
10x Genomics Visium captures the whole transcriptome on 55 µm spots. You get the H&E image,
the spot shapes and an expression table.

{{ line_Visium }}
:::

:::{grid-item-card} Visium HD
Visium HD reads the whole transcriptome on a gridded array, binned at 2, 8 or 16 µm
(the default table is `square_008um`). You get images, the bin shapes and one table per bin size.

{{ line_VisiumHD }}
:::

:::{grid-item-card} Xenium
Xenium is a targeted in-situ panel with subcellular resolution. You get individual transcripts
(points), cell and nucleus segmentation (shapes and labels), morphology images and a cell-by-gene table.

{{ line_Xenium }}
:::

:::{grid-item-card} Atera
Atera is a 10x preview platform that writes Xenium-format output, so it is read with the Xenium
reader and yields the same kinds of elements.

{{ line_Atera }}
:::

:::{grid-item-card} Spatial Transcriptomics
The original Spatial Transcriptomics method (ST, Ståhl et al. 2016, *Science*) predates Visium. It
profiles the whole transcriptome on 100 µm spots on ST 1k/2k arrays, giving images, spot shapes and a table.

{{ line_SpatialTranscriptomics }}
:::

::::

{{ other_assays }}
