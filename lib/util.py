import hashlib
import json
from typing import Any

_UNWANTED_FIELDS: list[str] = [
    "metadata",
    "start_line",
    "stop_line",
    "raw_meta",
    "raw_condition",
    "raw_strings",
    "rule_name",
    "tags"
]

def _hash_parsed_rule(rule: dict[str, Any]) -> str:
    hasher: Any = hashlib.new("sha1")
    rule_content: str = json.dumps(rule)
    hasher.update(rule_content.encode())

    return hasher.hexdigest()

def _remove_unwanted_fields(rule: dict[str, Any]) -> None:
    for field in _UNWANTED_FIELDS:
        if field not in rule: continue
        del rule[field]

    return
