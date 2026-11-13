/**
 * Author: Lukas Polacek, Simon Lindholm
 * Date: 2019-12-26
 * License: CC0
 * Source: folklore
 * Description:
 * Status:
 */

struct line {
	long long k, b;
	long long get(long long x) {
		return k * x + b;
	}
};

struct {
	line t[16 * N];
	int l[16 * N], r[16 * N];
	int root = 1, last = 2;

	void init(int v) {
		t[v].k = 0; t[v].b = 1e18; l[v] = r[v] = 0;
	}
	void update(int v, long long tl, long long tr, line x) {
		long long tm = (tl + tr) >> 1;
	
		if (t[v].get(tm) > x.get(tm)) {
			swap(t[v], x);
		}

		if (t[v].get(tl) > x.get(tl)) {
			if (!l[v]) l[v] = last++, init(l[v]);
			update(l[v], tl, tm, x);
		} else if (t[v].get(tr) > x.get(tr)) {
			if (!r[v]) r[v] = last++, init(r[v]);
			update(r[v], tm + 1, tr, x);
		}
	}
	long long get(int v, long long tl, long long tr, long long pos) {
		if (tl == tr) {
			return t[v].get(pos);
		}

		long long tm = (tl + tr) >> 1;
		if (pos <= tm) {
			if (!l[v]) return t[v].get(pos);
			return min(t[v].get(pos), get(l[v], tl, tm, pos));
		} else {
			if (!r[v]) return t[v].get(pos);
			return min(t[v].get(pos), get(r[v], tm + 1, tr, pos));
		}
	}
} T;
