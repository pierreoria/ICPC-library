/**
 * Author: Lukas Polacek, Simon Lindholm
 * Date: 2019-12-26
 * License: CC0
 * Source: folklore
 * Description:
 * Status: tested as part of DirectedMST.h
 */
#pragma once

struct DSU {
	vector<int> par, sz;
	vector<pair<int*, int>> history;
	
	DSU(int n) : par(n, -1), sz(n, 1) {}
	int time() { return (int)history.size(); }
	int get(int v) { return par[v] < 0 ? v : get(par[v]); }
	void rollback(int t) {
		while ((int)st.size() > t) {
			*history.back().first = history.back().second;
			history.pop_back();
		}
	}
	void join(int a, int b) {
		a = get(a), b = get(b);
		if (a == b) return;
		if (sz[a] < sz[b]) swap(a, b);
		history.push_back({&sz[a], sz[a]});
		history.push_back({&par[b], par[b]});
		sz[a] += sz[b];
		par[b] = a;
	}
}
