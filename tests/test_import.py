import sddb


def test_version_exposed():
    assert isinstance(sddb.__version__, str)
    assert sddb.__version__
