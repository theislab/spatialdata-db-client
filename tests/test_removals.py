import pytest


def test_removed_names_not_importable_from_sddb():
    import sddb

    for name in ("FrozenCohort", "PreflightReport", "TaskView"):
        assert not hasattr(sddb, name), f"{name} should be removed"


def test_removed_adapter_functions_gone():
    import sddb.adapter as a

    for name in ("build", "materialize", "check", "TaskView", "PreflightReport", "_resolve_elements"):
        assert not hasattr(a, name), f"adapter.{name} should be removed"


def test_freeze_module_gone():
    with pytest.raises(ModuleNotFoundError):
        import sddb.freeze  # noqa: F401


def test_new_names_importable():
    from sddb import CohortManifest, CohortDriftError, SpatialDataCohort, TaskAdapter  # noqa: F401
