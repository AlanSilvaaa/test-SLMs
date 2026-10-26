"""Read the repo-local environment before importing model/download libraries."""

from pathlib import Path


def load_repo_env(path=None):
    from dotenv import load_dotenv

    path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / ".env"
    # Do not interpret shell commands, interpolate other secrets, persist a Hub
    # login, or override variables already supplied by the caller's environment.
    load_dotenv(path, override=False, interpolate=False)
