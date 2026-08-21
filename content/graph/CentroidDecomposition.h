int getSizes(int u, int v){
    sizes[u] = 1;
    for(auto x: adj[u]){
        if(x != v && !dead[x]) sizes[u] += getSizes(x, u);
    }
    return sizes[u];
}
 
int getCentroid(int u, int v, int tot){
    for(auto x: adj[u]){
        if(x != v && !dead[x] && sizes[x] > tot/2) {
            return getCentroid(x, u, tot);
        }
    }
    return u;
}
 
int buildCTree(int v){
 
    int sz = getSizes(v, -1);
    int centroid = getCentroid(v, -1, sz);
 
    parC[centroid] = centroid;
    dead[centroid] = 1;
 
    for(auto x: adj[centroid]){
        if(!dead[x]){
            parC[buildCTree(x)] = centroid;
        }
    }
 
    return centroid;
}
