def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    results = {}
    for record in records:
        key = tuple(record[col] for col in key_columns)
        if key not in results.keys():
            results[key] = record
        else:
            if strategy == "first":
                continue
            elif strategy == "last":
                results[key] = record 
            else:
                prev_record = results[key]
                prev_cnt, cur_cnt = 0, 0
                for val1, val2 in zip(prev_record.values(), record.values()):
                    if val1 is None:
                        prev_cnt += 1
                    if val2 is None:
                        cur_cnt += 1
                if prev_cnt <= cur_cnt:
                    continue
                else:
                    results[key] = record
    return list(results.values())