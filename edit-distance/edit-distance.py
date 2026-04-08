def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    # Write code here
    N, M = len(s1), len(s2)
    if N == 0:
        return M
    if M == 0:
        return N

    dp = [[0 for col in range(N + 1)] for i in range(M + 1)]
    for col in range(N + 1):
        dp[0][col] = col
    for row in range(M + 1):
        dp[row][0] = row

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if s1[j - 1] == s2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[M][N]
            
    
    