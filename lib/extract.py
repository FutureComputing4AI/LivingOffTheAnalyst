from .parse import enum_yara_rules
from typing import Any

def _strings_to_rules(parsed_rules: dict[str, dict[str, Any]]) -> list[str]:
    rule_num: int = 0
    rules: list[str] = []

    for rule in parsed_rules.values():
        strings: list[str] = [string["value"] for string in rule.get("strings", list())]

        for string in strings:
            rule: str = "\t\t$s1 = %s\n" % string
            rule += "\n\tcondition:\n\t\t"
            rule += "$s1\n"
            rule += '}'
            rule = ("rule r%s {\n\tstrings:\n" % rule_num) + rule
            rule_num += 1
            rules.append(rule)

    return rules

# TODO: modules?
def strings_as_yara_rules(yara_rules_path: str, outfile: str) -> None:
    parsed_rules: dict[str, dict[str, Any]] = enum_yara_rules(yara_rules_path)

    # tmp
    for rule in parsed_rules.values():
        if rule.get("condition_terms"):
            del rule["condition_terms"]
        if rule.get("imports"):
            del rule["imports"]
        if rule.get("special_conditions"):
            del rule["special_conditions"]

    rules: list[str] = _strings_to_rules(parsed_rules)

    # remove indent in prod
    with open(outfile, 'w') as file:
        for rule in rules:
            file.write(rule + "\n\n")

    return
