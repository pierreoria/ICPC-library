const int MAXN = 1e5 + 5;
const int LG = 18;
int v[MAXN];
int m[MAXN][LG];
int LGS[MAXN];

int query(int l, int r){
    int sz = r - l + 1, lg = LGS[sz];
    return min(m[l][lg], m[r-(1<<lg)+1][lg]);
}

void solve(){
    int n; cin >> n;
    LGS[1] = 0;
    for(int i = 2; i<=n; i++) LGS[i] = LGS[i/2] + 1;
    
    for(int i = 0; i<n; i++){
        cin >> v[i];
        m[i][0] = v[i];
    }

    // pre processamento
    for(int l = 1; l<=LG; l++){
        for(int j = 0; j + (1 << l) - 1 < n; j++){
            m[j][l] = min(m[j][l-1], m[j + (1<<(l-1))][l-1]);
        }
    }

    // query 0-based
    int q, l, r; cin >> q;
    while(q--){
        cin >> l >> r;
        cout << query(l, r) << endll;
    }
}
