import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.asarray(predictions)
    def vote_for_single_sample(sample_votes):
        values, counts = np.unique(sample_votes, return_counts=True)
        most_vote_idx = np.argmax(counts)
        return values[most_vote_idx]
    results = np.apply_along_axis(vote_for_single_sample, axis=0, arr=predictions)
    return results.tolist()
        
        
        
        