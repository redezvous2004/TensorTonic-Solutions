def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    updated_features = []
    for request in requests:
        updated_ft = {**feature_store.get(request["user_id"], defaults), **request["online_features"]}
        updated_features.append(updated_ft)
    return updated_features
    