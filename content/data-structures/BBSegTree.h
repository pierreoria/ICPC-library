#include <bits/stdc++.h>

using namespace std;

#define br '\n'

const int mx = 8e5+5;
int n;
int t[mx];

// raiz: v = 1, tl = 0, tr = n-1
// intervalos fechados
// zero-based

void build(vector<int>& a, int v, int tl, int tr) {
    if (tl == tr) {
        t[v] = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
        t[v] = max(t[v*2] , t[v*2+1]);
    }
}

int lowerbound(int v, int tl, int tr, int l, int r, int x) {
    if(tl > r || tr < l) return -1;
    if(t[v] < x) return -1;

    if (tl == tr) return tl;

    int tm = tl + (tr-tl)/2;
    int left = lowerbound(2*v, tl, tm, l, r, x);
    if(left != -1) return left;
    return lowerbound(2*v+1, tm+1, tr, l ,r, x);
}

void update(int v, int tl, int tr, int pos, int new_val) {
    if (tl == tr) {
        t[v] = new_val;
    } else {
        int tm = (tl + tr) / 2;
        if (pos <= tm)
            update(v*2, tl, tm, pos, new_val);
        else
            update(v*2+1, tm+1, tr, pos, new_val);
        t[v] = max(t[v*2], t[v*2+1]);
    }
}

int main()
{
    ios::sync_with_stdio(false); cin.tie(NULL);

    int q,x;
    cin>>n>>q;

    vector<int> a(n);
    for (int i = 0; i < n; i++) cin>>a[i];
    build(a,1,0,n-1);
    
    for (int i = 0; i < q; i++)
    {
        cin>>x;
        int idx = lowerbound(1,0,n-1,0,n-1,x);
        cout << idx+1 << (i == q-1 ? '\n' : ' ');

        if (idx > -1) {
            a[idx] -= x;
            update(1,0,n-1,idx,a[idx]);
        }
    }



    return 0;
}
