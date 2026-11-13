/**
 * Author: chilli
 * License: CC0
 * Description: z[i]: abacaba -> 0010301.
 * Time: O(n)
 * Status: stress-tested
 */
#pragma once

vector<int> z(const string& s) {
	vector<int> z(s.size());
	int l = -1, r = -1;
	for (int i = 1; i < s.size(); ++i) {
		z[i] = i > r ? 0 : min(r - i + 1, z[i - l]);
		while (i + z[i] < s.size() && s[i + z[i]] == s[z[i]])
			++z[i];
		if (i + z[i] - 1 > r)
			l = i, r = i + z[i] - 1;
	}
	return z;
}
