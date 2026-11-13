/**
 * Author: Lukas Polacek
 * Date: 2009-10-30
 * License: CC0
 * Source: folklore/TopCoder
 * Description:
 * Status: Stress-tested
 */
#pragma once

struct F {
	int n; vector<long long> t1, t2;
	F(int m) : t1(m), t2(m), n(m) {}
	void update(int l, long long x) {
		if (l < 0) return;
		for (int i = l; i >= 0; i = (i & (i + 1)) - 1)
			t2[i] += x;
		for (int i = l | (l + 1); i < n; i |= i + 1) 
			t1[i] += x * (l - (i & (i + 1)) + 1);
	}
	long long get(int r) {
		if (r < 0) return 0; long long res = 0;
		for (int i = r; i >= 0; i = (i & (i + 1)) - 1)
			res += t1[i] + t2[i] * (i - (i & (i + 1)) + 1);
		for (int i = r | (r + 1); i < n; i |= i + 1)
			res += t2[i] * (r - (i & (i + 1)) + 1);
		return res;
	}
	void update(int l, int r, long long x) {
		update(l - 1, -x); update(r, x);
	}
	long long get(int l, int r) {
		return get(r) - get(l - 1);
	}
};
