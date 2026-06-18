import math
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    N = len(actual_tokens)
    cross_entropy = - sum(math.log(prob_distributions[i][actual_tokens[i]]) for i in range(N)) / N
    perplexity = math.exp(cross_entropy)
    return perplexity
        