/**
 * Author: Pierre
 * Date: 2026-08-21
 * License: CC0
 * Source: AI.Cash https://codeforces.com/blog/entry/18051
 * Description: 
 */
#pragma once

int t[2 * N];
void build(int n) {
	for (int i = 0; i < n; ++i)
		t[n + i] = a[i];
	for (int i = n - 1; i > 0; --i)
		t[i] = t[i << 1] + t[i << 1 | 1];
}

void modify(int pos, int val) {  // a[pos] := val
	for (t[pos += n] = val; pos > 1; pos >>= 1)
		t[pos >> 1] = t[pos] + t[pos ^ 1];
}

int query(int l, int r) {  // [l, r)
	int res = 0;
	for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
		if (l & 1) res += t[l++];
		if (r & 1) res += t[--r];
	}
	return res;
}
