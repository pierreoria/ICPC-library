/**
 * Author: Simon Lindholm
 * Date: 2019-12-28
 * License: CC0
 * Source: https://github.com/hoke-t/tamu-kactl/blob/master/content/data-structures/MoQueries.h
 * Description: 
 * Status: stress-tested
 */
#pragma once

const int C = 1000;
struct query {
	int l, r, t, it;
	bool operator<(const query& a) {
		if (l / C == a.l / C) {
			if (t / C == a.t / C)
				return r < a.r;
			return t < a.t;
		}
		return l < a.l;
	}
};
struct {
	int l = 1, r = 0, t = 0;
	vector<...> history; 

	void add(int pos) {}
	void del(int pos) {}

	void upd() {}
	void upd(int pt) {
		while (pt < t) { //-upd
			int it = --t; upd(hist[it]);
		}
		while (t < pt) { //+upd
			int it = t++; upd(hist[it]);
		}
	}

	int get() {}
	int get(int pl, int pr) {
		while (r < pr) 
			add(++r);
		while (l < pl)
			del(l++);
		while (l > pl)
			add(--l);
		while(r > pr)
			del(r--);
		return get();
	}
} mo;
