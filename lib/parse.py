import os
#import plyara
from . import util
from . import normalize
from typing import Any

# iterate over all of the .yara files in the provided directory (and subdirectories)
def enum_yara_rules(rules_dir: str) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}

    for root, _, files in os.walk(rules_dir):
        # if the file contains yara rules, store them
        for file in files:
            if not (file.endswith(".yara") or file.endswith(".yar")): continue

            relative_path: str = os.path.join(root, file)
            _parse_file(rules, relative_path)

    return rules

# utilize plyara to parse the rules in the provided file path
# this fxn also removes unnecessary fields (such as metadata),
#   as well as normalizing strings for our purposes
def _parse_file(parsed_rules: dict[str, dict[str, Any]], path: str) -> None:
    #parser: plyara.Plyara = plyara.Plyara()

    with open(path, 'r') as file:
        parsed_rules_in_file: list[dict[str, Any]] = parser.parse_string(file.read())

        # remove unwanted fields for each parsed rule in the file
        for rule in parsed_rules_in_file:
            util._remove_unwanted_fields(rule)

            if rule.get("strings"):
                normalize.normalize_strings(rule)

            # hash the file content to avoid duplicate rules
            rule_hash: str = util._hash_parsed_rule(rule)
            if rule_hash not in parsed_rules:
                parsed_rules[rule_hash] = rule

    return
