import pytest


@pytest.fixture(autouse=True)
def no_gpu_inventory_in_unit_tests(monkeypatch):
    monkeypatch.setattr("common.runtime.hardware_metadata", lambda: {"unit_test": True})


@pytest.fixture(autouse=True)
def no_real_credentials_in_cli_tests(monkeypatch):
    # Tests that exercise dotenv explicitly provide their own temporary file.
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")
    monkeypatch.setattr("main.load_repo_env", lambda: None)
