# API reference

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   sddb.Catalog
   sddb.Dataset
   sddb.SpatialDataCohort
   sddb.TaskAdapter
   sddb.TableAdapter
   sddb.Manifest
   sddb.CohortManifest
   sddb.SplitManifest
   sddb.CohortDriftError
   sddb.CatalogChangedError
   sddb.CatalogUnavailableError
   sddb.ManifestVersionMismatch
   sddb.open_sdata
   sddb.resolve
   sddb.table_centroids
   sddb.citations.citation_entries
```

## Loader collections

{mod}`sddb.loader` writes a cohort as a collection for the spatialbatch loader, and {mod}`sddb.contract` defines and
checks what the collection holds. Both need the `loader` extra.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   sddb.loader
   sddb.loader.materialize
   sddb.loader.default_chain
   sddb.loader.Rasterize
   sddb.loader.Centroids
   sddb.loader.TissueMask
   sddb.loader.StainNormalize
   sddb.loader.adapter_from_record
   sddb.contract
   sddb.contract.validate
   sddb.contract.validate_dataset
   sddb.contract.assert_valid
   sddb.contract.ContractError
```

{class}`~sddb.loader.LoaderItem` is a dataclass and the contract models are pydantic models; their fields are
described in each docstring.

```{eval-rst}
.. autoclass:: sddb.loader.LoaderItem
   :members: from_sdata

.. autoclass:: sddb.contract.DatasetContract

.. autoclass:: sddb.contract.SpatialEntry

.. autoclass:: sddb.contract.AdapterRecord

.. autoclass:: sddb.contract.Scalefactors

.. autoclass:: sddb.contract.SpatialMetadata
```
