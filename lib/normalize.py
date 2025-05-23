import random
import string as pystring  # oops
from typing import Any

_STRING_PREFIX_LENGTH: int = 16

# this fxn is responsible for anonymizing names of strings and byte sequences,
#   as well as matching string prefix groups with their condition_term equivalents
def normalize_strings(rule: dict[str, dict[str, Any]]) -> None:
    _rename_strings(rule)
    _normalize_rule(rule)
    _normalize_byte_sequences(rule)

    return

# adds fields to the rule if they don't exist
def _normalize_rule(rule: dict[str, dict[str, Any]]) -> None:
    if not rule.get("strings"):
        rule["strings"] = []
    if not rule.get("condition_terms"):
        rule["condition_terms"] = []
    if not rule.get("imports"):
        rule["imports"] = []

    rule["special_conditions"] = []
    return

# bytes have a ton of random newlines and space in them-- remove that
# text strings don't have quotes, for some reason
def _normalize_byte_sequences(rule: dict[str, dict[str, Any]]) -> None:
    for string in rule["strings"]:
        if string["type"] == "byte" and (string["type"][0] + string["type"][-1]) != "{}":
            sequence: str = string["value"][1:-1]
            sequence = ' '.join([line.strip() for line in sequence.splitlines()])
            string["value"] = '{' + sequence + '}'
        elif string["type"] == "text":
            string["value"] = '"' + string["value"] + '"'
        else: ...  # maybe something?

    return

def _rename_strings(rule: dict[str, dict[str, Any]]) -> None:
    # if a wildcard is present in the conditions, the strings must also have a matching prefix
    condition_wildcard_prefixes: dict[str, str] = {}
    for term in rule["condition_terms"]:
        if term.endswith('*'):
            new_prefix: list[str] = random.sample(pystring.ascii_letters, _STRING_PREFIX_LENGTH)
            condition_wildcard_prefixes[term] = ''.join(new_prefix)

    # anonymize string names in both the "strings" section and "condition_terms"
    strings: list[dict[str, Any]] = rule.get("strings", [])
    for i, string in enumerate(strings):
        new_name: str = ""

        # assign a new prefix to the strings in the rule
        # also for/else lol
        for old_prefix, new_prefix in condition_wildcard_prefixes.items():
            # chop off asterisk at the end of the old prefix
            if string["name"].startswith(old_prefix[:-1]):
                new_name = f"${new_prefix}_{i}"
                break
        else:
            new_name = ''.join(random.sample(pystring.ascii_letters, _STRING_PREFIX_LENGTH))
            new_name = f"${new_name}_{i}"

        # NOTE: am i tweaking? i swear to god i've used list.replace() before
        for j, term in enumerate(rule["condition_terms"]):
            if term == string["name"]:
                rule["condition_terms"][j] = new_name
            elif term in condition_wildcard_prefixes:
                rule["condition_terms"][j] = f"${condition_wildcard_prefixes[term]}_*"

        string["name"] = new_name

    return

