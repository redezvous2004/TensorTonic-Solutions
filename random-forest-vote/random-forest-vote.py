import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.asarray(predictions)
    results = []
    for i in range(predictions.shape[-1]):
        sample_votes = predictions[:, i]
        values, counts = np.unique(sample_votes, return_counts=True)
        most_vote_idx = np.argmax(counts)
        results.append(values[most_vote_idx])
    return results
        
        
        
        