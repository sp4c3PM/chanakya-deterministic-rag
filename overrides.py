"""
Expert Override Layer — Human-in-the-Loop grounding.
Corrections made by experts are stored as high-priority metadata,
overriding PDF-extracted values for all future queries.
"""
import json
import os
from datetime import datetime
from pathlib import Path

OVERRIDES_FILE = Path("overrides.json")

def load_overrides():
    if OVERRIDES_FILE.exists():
        with open(OVERRIDES_FILE) as f:
            return json.load(f)
    return {}

def save_override(entity, field, value, context, override_by="analyst"):
    overrides = load_overrides()
    key = f"{entity.lower()}.{field}"
    overrides[key] = {
        "value": value,
        "context": context,
        "override_by": override_by,
        "override_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f, indent=2)
    return overrides[key]

def get_override(entity, field):
    overrides = load_overrides()
    return overrides.get(f"{entity.lower()}.{field}")

def list_overrides():
    return load_overrides()
