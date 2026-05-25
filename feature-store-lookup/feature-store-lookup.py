def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    updated_features = []
    for request in requests:
        user_id = request["user_id"]
        if user_id not in feature_store.keys():
            updated_ft = {**defaults, **request["online_features"]}
        else:
            updated_ft = {**feature_store[user_id], **request["online_features"]}
        updated_features.append(updated_ft)
    return updated_features
    