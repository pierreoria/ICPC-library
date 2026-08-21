struct Edge {
    int u, v; ll sz;
    Edge(int u, int v, ll sz):u(u), v(v), sz(sz) {}
};

struct Dinic {
    int n, src, sink;
    vector<vector<int>>adj; 
    vector<Edge>alledges;
    vector<int>level, curp;

    Dinic (int n, int src, int sink) : n(n), src(src), sink(sink) {
        adj.resize(n);
    }

    void addEdge(int u, int v, ll sz){
        // u--; v--; # quando 1 indexado (lembrar de colocar n+1)
        adj[u].emplace_back(alledges.size());
        alledges.emplace_back(u, v, sz);
        adj[v].emplace_back(alledges.size());
        alledges.emplace_back(v, u, 0);
    }

    ll dfs(int u, ll f = 1e12){
        if(f == 0) return 0;
        if(u == sink) return f;

        for(auto &i = curp[u]; i < adj[u].size(); i++){
            int cur = adj[u][i], v = alledges[cur].v;
            if(level[u] + 1 == level[v]){
                ll qtd = dfs(v, min(f, alledges[cur].sz));
                if(qtd){
                    alledges[cur].sz -= qtd;
                    alledges[cur^1].sz += qtd;
                    return qtd;
                }
            }
        }

        return 0;
    }

    bool bfs(){
        level = vector<int>(n, n);
        level[src] = 0;
        queue<int>q;
        q.emplace(src);
        while(!q.empty()){
            int u = q.front(); q.pop();
            for(auto x: adj[u]){
                auto [edu, edv, edsz] = alledges[x];
                if(edsz == 0 || level[edv] != n) continue;
                level[edv] = level[u] + 1;
                q.emplace(edv);
            }
        }
        return level[sink] < n;
    }

    ll maxFlow(){
        ll tot = 0;
        while(bfs()){
            curp = vector<int>(n, 0);
            while(ll sz = dfs(src)) tot += sz;
        }
        return tot;
    }

    // verifica se o vertice u foi alcancado na ultima bfs
    bool check(int u){
        return level[u] < n;
    }
};
