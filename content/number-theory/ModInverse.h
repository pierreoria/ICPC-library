/**
 * Author: Simon Lindholm
 * Date: 2016-07-24
 * License: CC0
 * Source: Russian page
 * Description: Assumes LIM $\le$ mod and that mod is a prime.
 * Status: Works
 */
#pragma once

const ll mod = 1e9 + 7, LIM = 200000;
ll inv[LIM]; inv[1] = 1;
for (int i = 2; i < LIM; ++i)
    inv[i] = mod - (mod / i) * inv[mod % i] % mod;
