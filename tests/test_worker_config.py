"""The worker must listen on the queue tasks are actually routed to.

scripts/start_worker.sh passed --queues=default while celery_app routes
worker.tasks.* to the "celery" queue, so a script-started worker sat idle and
jobs piled up in the broker. Docker was unaffected because it starts the
worker with no --queues at all.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _routed_queues():
    from worker.celery_app import celery_app

    return {route["queue"] for route in celery_app.conf.task_routes.values()}


def test_tasks_are_routed_to_a_named_queue():
    assert _routed_queues() == {"celery"}


@pytest.mark.parametrize("script", ["start_worker.sh"])
def test_start_scripts_consume_the_routed_queue(script):
    path = ROOT / "scripts" / script
    if not path.is_file():
        pytest.skip(f"{script} not present")

    text = path.read_text(encoding="utf-8")
    listening = set(re.findall(r"--queues[= ]([A-Za-z0-9_,]+)", text))
    if not listening:
        return  # no --queues means the default queue, which Celery routes to

    consumed = {queue for spec in listening for queue in spec.split(",")}
    missing = _routed_queues() - consumed
    assert not missing, (
        f"{script} listens on {sorted(consumed)} but tasks are routed to "
        f"{sorted(_routed_queues())}; jobs would never be picked up"
    )


def test_registered_job_task_is_routed():
    import importlib

    # celery_app.include is lazy: the module registers its tasks on import,
    # which is what the worker does at boot and what submit() does on dispatch.
    importlib.import_module("worker.tasks")
    from worker.celery_app import celery_app

    assert "worker.tasks.run_registered_job" in celery_app.tasks
    pattern, route = next(iter(celery_app.conf.task_routes.items()))
    assert pattern == "worker.tasks.*"
    assert route["queue"] in _routed_queues()
