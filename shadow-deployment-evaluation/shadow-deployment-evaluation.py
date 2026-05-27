import math
def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    # Write code here
    n = len(production_log)
    if n == 0:
        return None
    
    production_accuracy = sum(1 for log in production_log if log["actual"] == log["prediction"]) / n
    shadow_accuracy = sum(1 for log in shadow_log if log["actual"] == log["prediction"]) / n
    accuracy_gain = shadow_accuracy - production_accuracy
    sorted_shadow = sorted(shadow_log, key=lambda x: x["latency_ms"])
    p95_idx = max(0, math.ceil(0.95 * n) - 1)
    p95_latency = sorted_shadow[p95_idx]["latency_ms"]
    agreement_rate = sum(1 for prod, shadow in zip(production_log, shadow_log) if prod["prediction"] == shadow["prediction"]) / n
    return {
        "promote": True if accuracy_gain >= criteria["min_accuracy_gain"] and p95_latency <= criteria["max_latency_p95"] and agreement_rate >= criteria["min_agreement_rate"] else False,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": p95_latency,
            "agreement_rate": agreement_rate
        }
    }