/**
 * Author: 
 * Description: 
 */

int st[N][LG];
void build() {
	for (int i = 0; i < n; ++i) st[i][0] = a[i];
	for (int j = 1; j < LG; ++j) {
		for (int i = 0; i + (1 << j) <= n; ++i) {
			st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
		}
	}
}

int get(int l, int r) {
	int lg = __lg(r - l + 1);
	return max(st[l][lg], st[r - (1 << lg) + 1][lg]);
}
