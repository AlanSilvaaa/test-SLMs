import pytest


@pytest.fixture(autouse=True)
def no_gpu_inventory_in_unit_tests(monkeypatch):
    monkeypatch.setattr("common.runtime.hardware_metadata", lambda: {"unit_test": True})
