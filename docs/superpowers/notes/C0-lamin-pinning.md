# C0 spike — lamin version pinning + zarr_url mutability (findings)

Probed the live catalog (608 rows) and a real store's root zarr attrs on 2026-10-07.

## Findings

1. **`zarr_url` is NOT version-qualified.** Example: catalog `uid = haOzFbJ7Bc9x7NW60001`
   (16-char stem `haOzFbJ7Bc9x7NW6` + 4-digit version ordinal `0001`), but
   `zarr_url = s3://scverse-spatial-eu-central-1/.lamindb/haOzFbJ7Bc9x7NW6.zarr` — the **stem only**,
   no version suffix. So the URL addresses the artifact's *current* materialization, effectively a
   "latest" pointer. A republish/reconversion can change the bytes at the same URL (matches the repo's
   known note "a store republished at the same URL is not detected").

2. **No explicit lamin-version field** in the catalog or the store attrs. The version lives only as the
   `uid` suffix (ordinal after the 16-char stem). Recording it is bookkeeping, but it **cannot be used
   to fetch a specific version** from `zarr_url` (which is stem-only).

3. **Strong drift signals DO live inside the store**, in the root group `.zattrs`:
   - `sddb_provenance`: `{sddb_id, canonical_key, study_id, container_version, converted_at}`
     (`converted_at` is a timestamp; `container_version` e.g. `loop-20260929` — both change on reconversion)
   - `spatialdata_attrs`: `{version, spatialdata_software_version}`
   - `spatialdata_io_software_version`, `spatialdata_io_reader`

## Ruling for Task 4 (objectmeta) and the freeze/verify design

- **Verify-on-read (fingerprint) is the operative guarantee**, not the version pin (the spec already says
  this; C0 confirms it is load-bearing, not belt-and-suspenders).
- `lamin_version`: record the `uid` version suffix when present (`uid` beyond the 16-char stem, else
  `None`) as bookkeeping. Do NOT rely on it to address a version at the URL.
- **Fold the root-group provenance into the fingerprint.** `compute_fingerprint` should hash, in addition
  to the element inventory + table list: the root `sddb_provenance` block (esp. `converted_at`,
  `container_version`) and `spatialdata_attrs`. These live in the store, so verify-on-read re-reads them
  and catches a reconversion even when element shapes are unchanged.
- `objectmeta.fetch_object_metadata` therefore also reads the root `.zattrs` (one metadata read, no array
  chunks) alongside `remote.elements(...)`.
