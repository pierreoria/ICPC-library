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
	vector<long long> t;
	F(int n) : t(n) {}
	void update(int l, long long val) { // a[l] += val
		for (; l < (int)t.size(); l |= l + 1) t[l] += val;
	}
	long long query(int r) { // sum of values in [0, r]
		long long res = 0;
		for (; r >= 0; r = (r & (r + 1)) - 1) res += t[r];
		return res;
	}
	long long query(int l, int r) { // sum of values in [l, r]
		return query(r) - query(l - 1);
	}
};
