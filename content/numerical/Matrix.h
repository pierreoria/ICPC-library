/**
 * Author: 
 * Description: 
 */

ll MOD = 1e9 + 7;

ll fexp(ll a, ll b, ll mod){
    ll ans = 1;
    while(b){
        if(b & 1) ans = (ans * a) % mod;
        a = (a*a) % mod;
        b >>= 1;
    }
    return ans;
}


template<typename T> struct Matrix {
	vector<vector<T>> mat;
	int n, m;

	Matrix(int N, int M=0) : n(N), m(M?M:N){ mat.assign(n, vector<T>(m, 0)); }

	friend Matrix operator* (const Matrix &a, const Matrix &b){
		assert(a.m == b.n);
		Matrix ans(a.n, b.m);
		for(int i=0; i<a.n; i++)
			for(int j=0; j<b.m; j++)
				for(int k=0; k<a.m; k++)
					ans.mat[i][j] += a.mat[i][k] * b.mat[k][j];
		return ans;
	}
};

Matrix fexp(Matrix a, ll n){
	int m = a.m;
	Matrix ans = Matrix(m);
	for(int i = 0; i < m; i++) ans.mat[i][i] = 1;
	while(n){
		if(n & 1) ans = ans * a;
		a = a * a;
		n >>= 1;
	}
	return ans;
}
