def retraining_policy(daily_stats, config):
    """
    Decide which days to trigger model retraining.
    """
    # Write code here
    retrain_days = []
    cooldown_trigger, day_since_retrain, used_budget = 0, 0, 0
    for stat in daily_stats:
        day_since_retrain += 1
        cooldown_trigger += 1
        if stat["drift_score"] > config["drift_threshold"] or stat["performance"] < config["performance_threshold"]:
            if (cooldown_trigger >= config["cooldown"] and used_budget < config["budget"]) or (len(retrain_days) == 0):
                retrain_days.append(stat["day"])
                used_budget += config["retrain_cost"]
                day_since_retrain = 0
                cooldown_trigger = 0
        else:
            if day_since_retrain == config["max_staleness"] and cooldown_trigger >= config["cooldown"] and used_budget < config["budget"]:
                retrain_days.append(stat["day"])
                used_budget += config["retrain_cost"]
                day_since_retrain = 0
                cooldown_trigger = 0
                
        if used_budget == config["budget"]:
            break
    return retrain_days
        