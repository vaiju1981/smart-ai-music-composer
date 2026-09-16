"""Valkey release gate per `docs/roadmap.md` §10 #9.

> Jobs: RQ + Valkey, bound locally, with RQ `JSONSerializer` and
> primitive job arguments. … an integration test against the selected
> Valkey release is a release gate.

The gate verifies, against a running Valkey:

1. the broker round-trips primitive values;
2. `enqueue_job` lands `run_job` on the `default` queue as a
   JSON-serialized payload (never pickle).

It self-skips when no Valkey is reachable, so the unit suite still
passes on a machine without one; the real-binary release run (which
starts Valkey) exercises it for real. Point `SAIMC_TEST_VALKEY_URL`
at a non-default host when testing a specific release.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

DEFAULT_TEST_VALKEY_URL = "valkey://127.0.0.1:6379/0"


def _valkey_url() -> str:
    return os.environ.get("SAIMC_TEST_VALKEY_URL") or DEFAULT_TEST_VALKEY_URL


def _valkey_reachable(url: str) -> bool:
    import redis

    from saimc.jobs.worker import redis_url

    try:
        client = redis.Redis.from_url(redis_url(url), socket_connect_timeout=1.0)
        client.ping()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _valkey_reachable(_valkey_url()),
    reason=f"no Valkey reachable at {_valkey_url()} (start one, or set SAIMC_TEST_VALKEY_URL)",
)


def test_broker_round_trips_primitive_values() -> None:
    import redis

    from saimc.jobs.worker import redis_url

    client = redis.Redis.from_url(redis_url(_valkey_url()))
    key = "saimc:release-gate:roundtrip"
    client.set(key, json.dumps({"job_id": "gate", "primitive": True}))
    assert json.loads(client.get(key)) == {"job_id": "gate", "primitive": True}
    client.delete(key)


def test_enqueue_job_lands_run_job_as_json(tmp_path: Path) -> None:
    import zlib

    import redis

    from saimc.jobs.storage import JobStorage
    from saimc.jobs.worker import enqueue_job, redis_url

    store = JobStorage(tmp_path)
    job = store.create("valkey gate")

    client = redis.Redis.from_url(redis_url(_valkey_url()))
    # Enqueue to a gate-only queue: a live worker on `default` (which
    # sits in an indefinite blocking BLMOVE) would drain the payload
    # before we can inspect it, and RQ's suspension flag cannot hold it
    # back. The gate queue has no listener, so the raw payload is stable.
    #
    # RQ 2.0 pushes only the job id onto the list (itself a primitive)
    # and stores the body in the job hash as zlib-compressed JSON
    # [func, instance, args, kwargs]. The gate asserts the body is
    # decompressible JSON — never pickle — with the primitive args.
    queue_key = "rq:queue:saimc:release-gate"
    enqueue_job(job.job_id, valkey_url=_valkey_url(), queue_name="saimc:release-gate")
    raw_job_id = client.lindex(queue_key, -1)
    assert raw_job_id is not None
    rq_job_id = raw_job_id.decode()
    body = zlib.decompress(client.hget(f"rq:job:{rq_job_id}", "data"))
    payload = json.loads(body)
    assert payload[0] == "saimc.jobs.worker.run_job"
    assert payload[2] == [job.job_id]
    client.delete(queue_key)
    client.delete(f"rq:job:{rq_job_id}")
