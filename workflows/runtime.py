"""Resolve declared workflow inputs and named outputs without mutating source data."""

from copy import deepcopy
from typing import Any, Dict, Iterable


def resolve_inputs(value: Any, inputs: Dict, outputs: Dict, declared: Iterable[str]):
    known = set(declared)

    def reference(name):
        if not isinstance(name, str):
            raise ValueError("Workflow reference must be a string")
        if name.startswith("inputs."):
            namespace, key = inputs, name[7:]
        elif name.startswith("outputs."):
            namespace, key = outputs, name[8:]
        elif name in outputs:
            namespace, key = outputs, name
        else:
            namespace, key = inputs, name
        if key in namespace:
            return deepcopy(namespace[key])
        current = namespace
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif (
                isinstance(current, list)
                and part.isdecimal()
                and int(part) < len(current)
            ):
                current = current[int(part)]
            else:
                raise ValueError(f"Unresolved workflow reference: {name}")
        return deepcopy(current)

    def resolve(item):
        if isinstance(item, dict):
            if set(item) == {"$ref"}:
                return reference(item["$ref"])
            if set(item) == {"$literal"}:
                return deepcopy(item["$literal"])
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        if isinstance(item, str) and (
            item in inputs or item in outputs or item in known
        ):
            return reference(item)
        return deepcopy(item)

    return resolve(value)


def bind_outputs(names, result):
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("Step outputs must be a list of nonempty names")
    if len(names) != len(set(names)):
        raise ValueError("Step output names must be unique")
    if not names:
        return {}
    if isinstance(result, dict) and all(name in result for name in names):
        return {name: deepcopy(result[name]) for name in names}
    if len(names) == 1:
        return {names[0]: deepcopy(result)}
    raise ValueError(f"Step result does not provide named outputs: {', '.join(names)}")
