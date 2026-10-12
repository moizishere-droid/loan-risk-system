"""Readiness polling without retrying prediction or deletion requests."""

import time

import requests


def wait_for_api(api_url, max_wait=300, retry_interval=2):
    """Wait for models AND database; retry connection errors during startup."""
    deadline = time.monotonic() + max_wait
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            response = requests.get(
                f"{api_url.rstrip('/')}/health", timeout=min(5, remaining)
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", {})
                if (data.get("status") == "healthy" and models
                        and all(value is True for value in models.values())):
                    return True
        except (requests.RequestException, ValueError, AttributeError):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(retry_interval, remaining))
