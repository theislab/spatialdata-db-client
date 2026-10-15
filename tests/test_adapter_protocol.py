from sddb.adapter import TaskAdapter


def test_protocol_is_runtime_checkable():
    class Fake:
        name = "fake"
        version = "1"
        config: dict = {}

        def requirements(self):
            return {}

        def validate_meta(self, meta):
            return []

        def validate(self, sdata):
            return []

        def build(self, sdata):
            return sdata

    assert isinstance(Fake(), TaskAdapter)
