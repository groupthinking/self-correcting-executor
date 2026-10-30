"""Bounded, explicit repair shared by imported and native workflow execution.

Repair functions are registered by the host, never imported from untrusted plans.
They must report a concrete change; the subsequent real execution proves whether
that change actually fixed the failure. A retry alone is not called a repair.
"""

from copy import deepcopy
from dataclasses import dataclass
import inspect


async def invoke(function, *args, **kwargs):
    result = function(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@dataclass
class Repair:
    inputs: dict
    before: object
    after: object
    reason: str

    def validate(self):
        if not isinstance(self.inputs, dict) or not self.reason:
            raise ValueError("Repair requires input mapping and reason")
        if self.before == self.after:
            raise ValueError("Repair did not report a concrete change")
        return self


class NonRetryableError(RuntimeError):
    """Execution/observation is uncertain; replay requires explicit reconciliation."""

    retryable = False


async def run_correcting(operation, inputs, *, attempts=1, repair=None, observe=None):
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError("attempts must be between 1 and 20")
    current, history = deepcopy(inputs), []

    async def record(entry):
        if observe is not None:
            try:
                await invoke(observe, deepcopy(entry))
            except Exception as error:
                entry["observation_error"] = str(error)
                return {
                    "status": "failed",
                    "execution_status": entry["status"],
                    "error": f"Observation failed; operation will not be replayed: {error}",
                    "attempts": history,
                    "output": entry.get("output"),
                }
        return None

    for attempt in range(1, attempts + 1):
        entry = {"attempt": attempt, "input": deepcopy(current)}
        error = None
        try:
            output = await invoke(operation, deepcopy(current))
            if isinstance(output, dict) and (
                output.get("success") is False
                or output.get("status") in {"failed", "error"}
            ):
                raise RuntimeError(output.get("error", "Operation reported failure"))
        except Exception as caught:
            error = caught
            entry.update(status="failed", error=str(error))
        else:
            # Never put observation or result-copy failures in the operation's
            # retry block: the operation has already returned successfully.
            try:
                entry.update(status="completed", output=deepcopy(output))
            except Exception as copy_error:
                entry.update(status="completed", observation_error=str(copy_error))
                history.append(entry)
                return {
                    "status": "failed",
                    "execution_status": "completed",
                    "error": "Successful output could not be captured; no replay",
                    "attempts": history,
                }
        history.append(entry)
        observation_failure = await record(entry)
        if observation_failure:
            return observation_failure
        if error is None:
            return {"status": "completed", "output": output, "attempts": history}
        if attempt == attempts or getattr(error, "retryable", True) is False:
            break
        if repair is not None:
            try:
                change = await invoke(repair, error, deepcopy(current), attempt)
                if not isinstance(change, Repair):
                    raise TypeError("Repair handler must return Repair")
                change.validate()
                entry["correction"] = {
                    "before": deepcopy(change.before),
                    "after": deepcopy(change.after),
                    "reason": change.reason,
                }
                current = deepcopy(change.inputs)
            except Exception as repair_error:
                entry["repair_error"] = str(repair_error)
                break
    return {"status": "failed", "error": history[-1]["error"], "attempts": history}


def restore_json_snapshot(path, approved_content):
    """Host-selected concrete repair for a damaged JSON input/artifact.

    A valid but different document is never overwritten. The host supplies the
    source snapshot and path; a workflow cannot choose arbitrary repair targets.
    """
    import json
    import os
    from pathlib import Path
    import tempfile

    target = Path(path).resolve()
    json.loads(approved_content)

    def repair(error, inputs, attempt):
        before = target.read_text()
        try:
            json.loads(before)
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError("Snapshot repair refuses to replace already-valid JSON")
        mode = target.stat().st_mode & 0o777
        fd, temporary = tempfile.mkstemp(prefix=".workflow-repair-", dir=target.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(approved_content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return Repair(
            inputs, before, target.read_text(), "Restore host-approved JSON snapshot"
        )

    return repair
