/**
 * Author: Noam527
 * Date: 2019-04-24
 * License: CC0
 * Source: folklore
 * Description:
 * Status: tested
 */
#pragma once

const ll mod = 1e9 + 7;
ll bpow(ll a, ll b) {
	ll ans = 1;
	for (; b; a = a * a % mod, b >>= 1)
		if (b & 1) ans = ans * a % mod;
	return ans;
}
