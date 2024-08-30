# DATA STRUCTURES

## BIT
/**
 * Description: 1-based range sums, point increments. lower bound in O(log n)
 */


const int mx = 4e6+6;
const int LOGN = 22; // teto(log(mx))
int n, bit[mx]; // 1-based: raiz é 1

int bit_search(int v) // busca por valor, não por range; equivalente a lower bound
{
    int sum = 0;
    int pos = 0;
    
    for(int i = LOGN; i>=0; i--) 
    {
        if(pos + (1 << i) <= n and sum + bit[pos + (1 << i)] < v) // <= n, n numero de elementos
        {
            sum += bit[pos + (1 << i)];
            pos += (1 << i);
        }
    }

    return pos + 1; // +1 because 'pos' will have position of largest value less than 'v'
}

// incremento, NÃO é equivalente a arr[idx] = v. guardar array original se preciso
void increment(int idx, ll v) 
{ 
    while(idx <= n) {
        bit[idx] += v;
        idx += idx & -idx;
    }
}

int query (int idx) 
{ 
    int ans = 0;
    while(idx > 0) {
        ans += bit[idx];
        idx -= idx & -idx;
    }

    return ans;
}


## Lazy Seg

/**
 * Description: Range assignment, point and range queries segment tree. 
 * 0-based, closed intervals. root at index 1, responsible for array[0:n-1]
 */


struct segTree{
    int n;
    vector<int> val, sum, lazy; 
    // pode ter mais atributos.
    // tb pode ser vetor de nós, onde nó é struct definida separadamente

    // segTree seg(n);
    segTree(int nn)
    {
        n = nn;
        val.resize(n<<2);
        sum.resize(n<<2);
        lazy.resize(n<<2);
    }

    // chama seg.build(a,1,0,n-1);
    void build(vector<int>& a, int node, int l, int r)
    {
        if (l == r) {
            val[node] = a[l];
        }
        else {
            int m = (l+r)>>1;
            build(a, node<<1, l, m);
            build(a, (node<<1)|1, m+1 ,r);
            val[node] = val[node<<1] + val[(node<<1)|1];
            // mais genericemente, comb(l,r)
        }
    }

    // atualiza nó individual
    void update_lazy(int node, int l, int r)
    {
        // 0 nesse caso é impossível como valor, significa. mudar em problema q 0 pode ocorrer
        if (!lazy[node]) return; // pra garantir mesmo quando é chamado pra nó errado
        int v = lazy[node]; // lazy não é só flag, guarda valor a ser propagado
        val[node] = v;
        sum[node] = factorsum(v) * (r-l+1); // específico do problema
    }

    // só é chamado para nós com filhos, propaga
    void push(int node, int l, int r)
    {
        // novamente, se 0 for valor válido, trocar isso
        if (lazy[node])
        {
            lazy[node<<1] = lazy[(node<<1)|1] = lazy[node];
            update_lazy(node, l, r);

            int m = (l+r)>>1;
            update_lazy(node<<1, l, m);
            update_lazy((node<<1)|1, m+1, r);
            lazy[node] = 0;
        }
    }

    void assign_range(int node, int tl, int tr, int l, int r, int v)
    {
        if (l > r) return;
        if (tl == l && tr == r) {
            lazy[node] = v;
            if (tl == tr) update_lazy(node, tl, tr);
            else push(node, tl, tr);
        }
        else {
            push(node, tl, tr);
            int tm = (tl + tr)>>1;
            assign_range(node<<1, tl, tm, l, min(tm,r), v);
            assign_range((node<<1)|1, tm+1, tr, max(tm+1,l), r, v);

            sum[node] = sum[node<<1] + sum[(node<<1)|1];
            // mais genericamente, comb(l,r)
        }
    }

    int range_query(int node, int tl, int tr, int l, int r)
    {
        if (l > r) return 0; // valor que não vai afetar query
        if (tl == l && tr == r) {
            if (tl == tr) update_lazy(node, tl, tr);
            else push(node, tl, tr);
            return sum[node];
        }
        push(node, tl, tr);
        int tm = (tl + tr)>>1;
        return sumquery(node<<1, tl, tm, l, min(r,tm)) +
               sumquery((node<<1)|1, tm+1, tr, max(tm+1,l), r);
        // mais genericamente, comb(l,r)
    }

    int point_query(int node, int tl, int tr, int pos)
    {
        if (tl > tr) return 0;
        if (tl == pos && tr == pos){
            update_lazy(node, tl, tr);
            return val[node];
        }
        push(node, tl, tr);
        int tm = (tl + tr)>>1;
        // bifurca: não pega os 2 filhos
        if (pos <= tm)
            return point_query(node<<1, tl, tm, pos);
        return point_query((node<<1)|1, tm+1, tr, pos);
    }

};

## Range update seg

/**
 * Description: vanilla range update seg, sum point queries
 0-based
 root = index 1 (tl = 0, tr = n-1)
 closed intervals -> [l,r] type queries
/*
 
*/
#pragma once

const int mx = 1e5+5;
int t[4*mx];

void build(vector<int>& a, int v, int tl, int tr) {
    if (tl == tr) {
        t[v] = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
        t[v] = 0;
    }
}

// range update: adds 'add' to each element in range, DOES NOT set each element to 'add'
void update(int v, int tl, int tr, int l, int r, int add) {
    if (l > r)
        return;
    if (l == tl && r == tr) {
        t[v] += add;
    } else {
        int tm = (tl + tr) / 2;
        update(v*2, tl, tm, l, min(r, tm), add);
        update(v*2+1, tm+1, tr, max(l, tm+1), r, add);
    }
}

// POINT query: single element
int get(int v, int tl, int tr, int pos) {
    if (tl == tr)
        return t[v];
    int tm = (tl + tr) / 2;
    if (pos <= tm)
        return t[v] + get(v*2, tl, tm, pos);
    else
        return t[v] + get(v*2+1, tm+1, tr, pos);
}

## Seg iterativa

/**
 * Description: low constant factors, 2*n memory iterative seg tree
  0-based, [l,r) type queries
  seg[n] to seg[2n-1] corresponds to array[0] to array[n-1]
  for N != 2^k, cannot be used as is for efficient binary search, as there will be more than one root 
 */

const int N = 1e5;  // limit for array size
int n;  // array size
int t[2 * N];

void build() {  // build the tree (first set positions n through 2n-1 as original array
  for (int i = n - 1; i > 0; --i) t[i] = t[i<<1] + t[i<<1|1];
}

void modify(int p, int value) {  // set value at position p
  for (t[p += n] = value; p > 1; p >>= 1) t[p>>1] = t[p] + t[p^1];
}

int query(int l, int r) {  // sum on interval [l, r)
  int res = 0;
  for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
    if (l&1) res += t[l++];
    if (r&1) res += t[--r];
  }
  return res;
}


## Matriz

const int MOD=1e9+7;
const int ms =2;
 
class Matrix{
 public:
   ll mat[ms][ms] = {{0,0},{0,0}};
   //vector<vector<ll>> mat= vector<vector<ll>>(n,vector<ll>(n));
 
   Matrix operator * (const Matrix &p){
 
     Matrix ans;
     for(int i = 0; i<ms; i++){ 
       for(int j = 0; j<ms; j++){
         for(int k = ans.mat[i][j] = 0; k<ms;k++){
         
           ans.mat[i][j] = ((ans.mat[i][j] + 1LL * (mat[i][k]%MOD) *(p.mat[k][j]%MOD)))%MOD;
 
         }
       }
     }
     return ans;
 
   }
};
 
Matrix mfxp(Matrix a, ll b){
 
 Matrix ans;
 for(int i=0;i<ms;i++){
   ans.mat[i][i] = 1;
 }
 
 while(b){
   if(b&1) ans = ans*a;
   a = a*a;
   b>>=1; 
 }
 
 return ans;
 
}


# Geometria

## Point

const double inf = 1e100, eps = 1e-9;
const long double PI = acos(-1.0L);
int cmp(double a, double b = 0){
    if(abs(a-b) < eps)return 0;
    return (a<b?-1:1);
}

struct Point{
    int x, y;
    Point(double x=0, double y=0) : x(x), y(y){}
    Point(const Point& p):x(p.x), y(p.y){}
    bool operator < (const Point& p) const{
        if(cmp(x, p.x) != 0)return x<p.x;
        return cmp(y, p.y) < 0;
    }
    bool operator == (const Point& p) const{
        if(cmp(x, p.x) == 0 && cmp(y, p.y) == 0)return true;
        return false;
    }
    bool operator != (const Point& p)const {return !(p == *this);}
    Point operator + (const Point& p) const {return Point(x+p.x,y+p.y);}
	Point operator - (const Point& p) const {return Point(x-p.x,y-p.y);}
	Point operator * (const double k) const {return Point(x*k,y*k);}
	Point operator / (const double k) const {return Point(x/k,y/k);}
};

double dot (const Point& p,const Point& q) { return p.x*q.x + p.y*q.y; }
double cross (const Point& p,const Point& q) { return p.x*q.y - p.y*q.x; }
double norm(const Point& p) { return hypot(p.x,p.y); }
double dist(const Point& p, const Point& q) { return hypot(p.x-q.x,p.y-q.y); }
double dist2(const Point& p, const Point& q) { return dot(p-q,p-q); }
Point perp(const Point& p) { return Point(-p.x,p.y); }
Point normalize(const Point &p) { return p/hypot(p.x, p.y); }
double angle (const Point& p, Point& q) { return atan2(cross(p, q), dot(p, q)); }
long double angle (const Point& p) {
    if(p.x < 0 && p.y == 0)return PI;
    else if(p.x == 0 && p.y == 0)return 0;
    return atan2l(p.y, p.x); 
}

## Min enclosing circle
/**
 * Description: Computes the minimum circle that encloses a set of points.
 * Time: expected O(n)
 * Status: stress-tested
 */

double ccRadius(P& A, P& B, P& C) {
  return len(B, A)*len(C,B)*len(A,C)/
      abs(cross((B-A), (C-A)))/2.0;
}
 
P ccCenter(P& A, P& B, P& C) {
  P b = C-A, c = B-A;
  return A + perp(b*dist2(c)-c*dist2(b))/cross(b, c)/2;
}
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
pair<P, double> mec(vector<P>& pts){
	shuffle(begin(pts),end(pts),rng);
	P o = pts[0];
	const double EPSS = 1+1e-8;
	double r = 0;
	for(int i = 0; i < pts.size(); i++) if(len(o, pts[i]) > r * EPSS){
		o = pts[i], r = 0;
		for(int j = 0; j < i; j++) if(len(o, pts[j]) > r * EPSS){
			o = (pts[i]+pts[j])/2.0;
			r = len(o, pts[i]);
			for(int k = 0; k < j; k++) if(len(o, pts[k]) > r * EPSS){
				o = ccCenter(pts[i],pts[j],pts[k]);
				r = len(o, pts[i]);
			}
		}
	}
	return {o, r};
}

# Graph


## 2-sat
/**
 * Description: Satisfiability of boolean statement in linear time. Supports or,xor,eq,and clauses.
 * constructs valid assignment if needed.
 * Status: Tested on recent GYM problem
 */



struct two_sat {
    int n;
    vector<vector<int>> g, gr; // gr is the reversed graph
    vector<int> comp, topological_order, answer; // comp[v]: ID of the SCC containing node v
    vector<bool> vis;
 
    two_sat() {}
 
    two_sat(int _n) { init(_n); }
 
    void init(int _n) {
        n = _n;
        g.assign(2 * n, vector<int>());
        gr.assign(2 * n, vector<int>());
        comp.resize(2 * n);
        vis.resize(2 * n);
        answer.resize(2 * n);
    }
 
    void add_edge(int u, int v) {
        g[u].push_back(v);
        gr[v].push_back(u);
    }
 
    // For the following three functions
    // int x, bool val: if 'val' is true, we take the variable to be x. Otherwise we take it to be x's complement.
 
    // At least one of them is true
    void add_clause_or(int i, bool f, int j, bool g) {
        add_edge(i + (f ? n : 0), j + (g ? 0 : n));
        add_edge(j + (g ? n : 0), i + (f ? 0 : n));
    }
 
    // Only one of them is true
    void add_clause_xor(int i, bool f, int j, bool g) {
        add_clause_or(i, f, j, g);
        add_clause_or(i, !f, j, !g);
    }
 
    // Both of them have the same value
    void add_clause_eq(int i, bool f, int j, bool g) {
        add_clause_xor(i, !f, j, g);
    }

    void add_clause_and(int i, bool f, int j, bool g) {
        add_clause_or(i,f,i,f);
        add_clause_or(j,g,j,g);
    }
 
    // Topological sort
    void dfs(int u) {
        vis[u] = true;
 
        for (const auto &v : g[u])
            if (!vis[v]) dfs(v);
 
        topological_order.push_back(u);
    }
 
    // Extracting strongly connected components
    void scc(int u, int id) {
        vis[u] = true;
        comp[u] = id;
 
        for (const auto &v : gr[u])
            if (!vis[v]) scc(v, id);
    }
 
    // Returns true if the given proposition is satisfiable and constructs a valid assignment
    bool satisfiable() {
        fill(vis.begin(), vis.end(), false);
 
        for (int i = 0; i < 2 * n; i++)
            if (!vis[i]) dfs(i);
 
        fill(vis.begin(), vis.end(), false);
        reverse(topological_order.begin(), topological_order.end());
 
        int id = 0;
        for (const auto &v : topological_order)
            if (!vis[v]) scc(v, id++);
 
        // Constructing the answer
        for (int i = 0; i < n; i++) {
            if (comp[i] == comp[i + n]) {
                return false;
            }
            answer[i] = (comp[i] > comp[i + n] ? 1 : 0);
        }
 
        return true;
    }
};


## Binary Lifting

/**
 * Description: Calculate power of two jumps in a tree,
 * to support fast upward jumps and LCAs.
 * This implementation includes max and min edges between node and 2^k-th ancestor
 * Time: construction $O(N \log N)$, queries $O(\log N)$
 * Status: tested
 */
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

## Kosaraju

/**
 * Description: Finds strongly connected components in a
 * directed graph. If vertices $u, v$ belong to the same component,
 * we can reach $u$ from $v$ and vice versa.
 * Time: O(E + V)
 * Status: tested on Iudex problem, recent codeforces Gym contest (não, sei usar vírgula)
 */
#pragma once

bitset<mx> vis,repvis;

void dfs(vector<vector<int>>& adj, stack<int>& ord, int u)
{
    vis[u] = 1;
    for (int v : adj[u]) {
        if (!vis[v]) {
            dfs(adj,ord,v);
        }
    }
    ord.push(u);
}

// s (source) eh representante do scc
void dfs2(vector<vector<int>>& scc, vector<vector<int>>& adjT, vector<int>& rep, int s, int u) 
{
    vis[u] = 1;
    rep[u] = s;
    scc[s].pb(u);

    for (int v : adjT[u]) {
        if (!vis[v]) {
            dfs2(scc,adjT,rep,s,v);
        }
    }
}

int main() 
{
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int t,n,u,g,v,r,sz,repcnt;
    char p;

    cin>>t;

    while(t--)
    {
        cin>>n;

        vis.reset();
        repvis.reset();
        vector<vector<int>> adj(n);
        vector<vector<int>> adjT(n);
        vector<vector<int>> scc(n);
        vector<int> rep(n);
        stack<int> ord;
        repcnt = 0;
        string gp; // numero:

        ord.push(-1); // evitar verificacao empty


        for (int i = 0; i < n; i++)
        {
            cin>>u>>gp;

            gp.pop_back(); // dois pontos

            g = stoi(gp);

            for (int j = 0; j < g; j++)
            {
                cin>>v;
                adj[u].pb(v);
                adjT[v].pb(u);
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (!vis[i]) {
                dfs(adj,ord,i);
            }
        }

        vis.reset();

        while (ord.top() != -1)
        {
            u = ord.top(); ord.pop();
            if (!vis[u]) {
                repcnt++;
                dfs2(scc,adjT,rep,u,u);
            }
        }

        cout << repcnt << br;
        for (int i = 0; i < n; i++)
        {
            r = rep[i];
            if (!repvis[r])
            {
                repvis[r] = 1;
                sz = scc[r].size();
                sort(scc[r].begin(), scc[r].end());
                for (int j = 0; j < sz; j++){
                    cout << scc[r][j] << (j == sz-1 ? '\n' : ' ');
                }
            }
        }
        cout << br;
    }

    return 0;
}

# Number Theory

# Diofantina + Euclides Extendido

int gcd_ext(int a, int b, int& x, int &y) {
  if (b == 0) {
    x = 1, y = 0;
    return a;
  }
  int nx, ny;
  int gc = gcd_ext(b, a % b, nx, ny);
  x = ny;
  y = nx - (a / b) * ny;
  return gc;
}

vector<int> diophantine(int D, vector<int> l) {
  int n = l.size();
  vector<int> gc(n), ans(n);
  gc[n - 1] = l[n - 1];
  for (int i = n - 2; i >= 0; i--) {
    int x, y;
    gc[i] = gcd_ext(l[i], gc[i + 1], x, y);
  }
  if (D % gc[0] != 0) {
    return vector<int>();
  }
  for (int i = 0; i < n; i++) {
    if (i == n - 1) {
      ans[i] = D / l[i];
      D -= l[i] * ans[i];
      continue;
    }
    int x, y;
    gcd_ext(l[i] / gc[i], gc[i + 1] / gc[i], x, y);
    ans[i] = (long long int) D / gc[i] * x % (gc[i + 1] / gc[i]);
    if (D < 0 && ans[i] > 0) {
      ans[i] -= (gc[i + 1] / gc[i]);
    }
    if (D > 0 && ans[i] < 0) {
      ans[i] += (gc[i + 1] / gc[i]);
    }
    D -= l[i] * ans[i];
  }
  return ans;
}

