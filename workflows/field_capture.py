"""Validate caller-supplied WRKFLW records without inventing field data or risk rules."""

from copy import deepcopy
from datetime import date

COLLECTIONS = ("daily_logs", "tasks", "rfis")


def normalize_field_capture(capture):
    if not isinstance(capture, dict):
        raise ValueError("field_capture must be a project record mapping")
    project_id = capture.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise ValueError("field_capture requires project_id")
    report_date = capture.get("report_date")
    if not isinstance(report_date, str):
        raise ValueError("field_capture requires an ISO report_date")
    date.fromisoformat(report_date)
    sources = []
    records = {}
    for collection in COLLECTIONS:
        items = capture.get(collection, [])
        if not isinstance(items, list):
            raise ValueError(f"{collection} must be a list")
        records[collection] = deepcopy(items)
        seen = set()
        for record in items:
            if not isinstance(record, dict):
                raise ValueError(f"{collection} records must be mappings")
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError(f"{collection} requires unique nonempty record IDs")
            seen.add(identifier)
            for key in ("project_id", "projectId"):
                if key in record and record[key] != project_id:
                    raise ValueError(
                        "Field capture contains a record from another project"
                    )
            sources.append(
                {
                    "source_id": f"{collection}/{identifier}",
                    "collection": collection,
                    "record_id": identifier,
                    "project_id": project_id,
                    "record": deepcopy(record),
                }
            )
    if not sources:
        raise ValueError("Provide at least one daily log, task or RFI record")
    return {
        "success": True,
        "action": "data_processing",
        "kind": "field_capture",
        "project_id": project_id,
        "report_date": report_date,
        "records": records,
        "sources": sources,
        "capture": deepcopy(capture),
    }
