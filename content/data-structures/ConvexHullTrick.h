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
	long double get(long double x) {
		return k * x + b;
	}
};
long double ist(line a, line b) {
	return (long double)(b.b - a.b) / (a.k - b.k);
}

vector<line> cht;
vector<long double> pt;

void add(line x) {
	while (cht.size() >= 2 && cht.back().get(pt.back()) > x.get(pt.back())) {
		cht.pop_back(); pt.pop_back();
	}

	if (cht.empty()) {
		pt.push_back(-inf);
	} else {
		pt.push_back(ist(cht.back(), x));
	}

	cht.push_back(x);
};

long double get(long long x) {
	int it = upper_bound(pt.begin(), pt.end(), x) - pt.begin() - 1;
	return cht[it].get(x);
};
