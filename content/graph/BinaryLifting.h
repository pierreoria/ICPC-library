/**
 * Author: Pierre Oriá
 * Date: 2024-08-30
 * License: CC0
 * Source: Folklore
 * Description: Calculate power of two jumps in a tree,
 * to support fast upward jumps and LCAs.
 * This implementation includes max and min edges between node and 2^k-th ancestor
 * Time: construction $O(N \log N)$, queries $O(\log N)$
 * Status: tested
 */
#pragma once
#include <bits/stdc++.h>

using namespace std;

#define br '\n'
#define pb push_back

typedef tuple<int,int> ii;

const int inf = 1e9;
const int MXN = 1e5+5;
const int lg = 21;
int cnt = 0;
int in[MXN],out[MXN],par[MXN][lg], mx[MXN][lg], mn[MXN][lg], h[MXN];
bitset<MXN> vis;

void dfs(vector<vector<ii>>& adj, int u, int p,  int height, int w)
{
	vis[u] = 1;
	in[u] = cnt++;
	h[u] = height;
	if (u) mx[u][0] = mn[u][0] = w, par[u][0] = p;
	
	for (ii wv: adj[u]){
	    int w1 = get<0>(wv);
	    int v = get<1>(wv);
	    if (!vis[v])
	    {
		dfs(adj, v, u, height+1,w1);
	    }
	}
	
	out[u] = cnt++;
	}
	
	bool isancestor(int a, int b) // a ancestral de b
	{
	return (in[a] <= in[b] && out[a] >= out[b]);
	}
	
	void build(int n)
	{
	for (int j = 1; j < lg; j++)
	{
	    for (int i = 0; i < n; i++) {
		int a = par[i][j-1];
		if (a != -1 && par[a][j-1] != -1){
		    par[i][j] = par[a][j-1];
		    mx[i][j] = max(mx[i][j-1], mx[a][j-1]);
		    mn[i][j] = min(mn[i][j-1], mn[a][j-1]);
		}
	    }
	}
}

int lca(int u, int v)
{
if (isancestor(u,v)) return u;
if (isancestor(v,u)) return v;
for (int i = 20; i >= 0; i--){
    if (par[u][i] != -1 && !isancestor(par[u][i],v)){
	u = par[u][i];
    }
}
return par[u][0];
}

// retornar nível h do lca, fazer bb em u e v até ancestral de nível h
ii solve(int u, int v)
{
int l = lca(u,v);

int lvl = h[l];
int ansmn = inf;
int ansmx = 0;

// MXLOG might be != 20
for (int i = 20; i >= 0; i--){
    if (par[u][i] != -1 && h[par[u][i]] > lvl){
	ansmn = min(ansmn, mn[u][i]);
	ansmx = max(ansmx, mx[u][i]);
	u = par[u][i];
    }
}

if (u != l) {
    ansmn = min(ansmn, mn[u][0]);
    ansmx = max(ansmx, mx[u][0]);
}

for (int i = 20; i >= 0; i--){
    //cout << "par[" << v << "][" << i << "]: " << par[v][i] << endl;
    if (par[v][i] != -1 && h[par[v][i]] > lvl){
	//cout << "h[ "<< par[v][i] << "]: " << h[par[v][i]] << endl;
	ansmn = min(ansmn, mn[v][i]);
	ansmx = max(ansmx, mx[v][i]);
	v = par[v][i];
    }
}

//cout << "u:  " << u << " v: " << v << " l: " << l << endl;

if (v != l){
    ansmn = min(ansmn, mn[v][0]);
    ansmx = max(ansmx, mx[v][0]);
}

return {ansmn, ansmx};
}



int main()
{
ios::sync_with_stdio(false); cin.tie(NULL);

memset(mn,63, sizeof(mn));
memset(mx,-1, sizeof(mx));
memset(par,-1, sizeof(par));

int n; cin>>n;

int u,v,w;
vector<vector<ii>> adj(n);

for (int i = 0; i < n-1; i++){
    cin>>u>>v>>w; u--; v--;
    adj[u].pb({w,v});
    adj[v].pb({w,u});
}

dfs(adj, 0, 0, 0,0);

build(n);

int q; cin>>q;

while(q--)
{
    cin>>u>>v; u--; v--;
    cout << get<0>(solve(u,v)) << " " << get<1>(solve(u,v)) << br;
}

}  
