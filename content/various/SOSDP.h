/**
 * Author: 
 * Description: 
 */

const int N = 20;
ll dp[1<<N], iVal[1<<N];

void sosDP(){ // O(N * 2^N) 
    for(int i=0; i<(1<<N); i++) 
        dp[i] = iVal[i];

	for(int i=0; i<N; i++)
		for(int mask=0; mask<(1<<N); mask++)
			if(mask&(1<<i))
				dp[mask] += dp[mask^(1<<i)];
}
