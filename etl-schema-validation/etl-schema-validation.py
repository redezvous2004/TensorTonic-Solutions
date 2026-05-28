def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    # Write code here
    results = []
    schema_dict = {
        item["column"]: {k : v for k, v in item.items() if k != "column"}
        for item in schema
    }
    for i, record in enumerate(records):
        check_result = []
        for key, condition in schema_dict.items():
            if key not in record.keys():
                check_result.append(f"{key}: missing")
            else:
                val = record[key]
                if val is None:
                    if not condition["nullable"]:
                        check_result.append(f"{key}: null")
                    continue
                if condition["type"] == "float":
                        accept_type = ["float", "int"]
                else:
                    accept_type = [condition["type"]]  
                if type(val).__name__ not in accept_type:
                    check_result.append(f"{key}: expected {condition["type"]}, got {type(val).__name__}")
                    continue
                if "min" in condition.keys() and val < condition["min"]:
                    check_result.append(f"{key}: out of range")
                if "max" in condition.keys() and val > condition["max"]:
                    check_result.append(f"{key}: out of range")
        is_valid = len(check_result) == 0
        results.append((i, is_valid, check_result))
    return results
                                