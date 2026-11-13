/**
 * Author: Johan Sannemo
 * Date: 2016-12-15
 * License: CC0
 * Description: p[i]: abacaba -> 0010123.
 * Time: O(n)
 * Status: Tested on kattis:stringmatching
 */
#pragma once

vector<int> p(const string& s) {
	vector<int> p(s.size());
	for (int i = 1; i < s.size(); ++i) {
		int g = p[i - 1];
		while (g && s[i] != s[g]) g = p[g - 1];
		p[i] = g + (s[i] == s[g]);
	}
	return p;
}
