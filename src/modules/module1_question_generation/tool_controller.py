from tools.tools import *
def verify_deterministic_assertions(llm_output, assertions_schema):
    """
    Takes LLM output and an assertions schema. Runs checks based on schema types
    against the LLM output and returns results.
    """
    results = {}
    try:
        data = assertions_schema
        deterministic_checks = data.get("deterministic", [])
        for item in deterministic_checks:
            check_type = item['check_type']
            value = item["value"]
            if check_type == "regex":
                results[f"Regex format - `{value}`"] = "Satisfied" if verify_regex(llm_output, value) else "Failed"
            elif check_type == "json-format":
                results[f"Json format - `{value}`"] =  "Satisfied" if verify_json_format(value) else "Failed"
            elif check_type == "contains":
                results[f"Contains - `{value}`"] =  "Satisfied" if verify_contains(llm_output, value) else "Failed"
            else:
                results[f"unknown-tool:{check_type}"] = False

    except json.JSONDecodeError:
        return {"error": "Invalid JSON in assertions schema"}
    # print("Assertion results", results, data, deterministic_checks)
    return results
