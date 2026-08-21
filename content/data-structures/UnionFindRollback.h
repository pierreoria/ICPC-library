/**
 * Author: Lukas Polacek, Simon Lindholm
 * Date: 2019-12-26
 * License: CC0
 * Source: folklore
 * Description: Disjoint-set data structure with undo.
 * If undo is not needed, skip st, time() and rollback().
 * Usage: int t = uf.time(); ...; uf.rollback(t);
 * Time: $O(\log(N))$
 * Status: tested as part of DirectedMST.h
 */
#pragma once

struct RollbackUF {
    vector<int> e; vector<pair<int, int>> st;

    RollbackUF(int n) : e(n, -1) {}

    int size(int x) { return -e[find(x)]; }
    
    int find(int x) { return e[x] < 0 ? x : find(e[x]); }
    
    int time() { return (int)(st.size()); }
    
    void rollback(int t) {
        while(st.size() > t){
            auto [u, v] = st.back(); st.pop_back();
            e[u] = v;
        }    
    }

    bool join(int a, int b) {
        a = find(a), b = find(b);
        if (a == b) return false;
        if (e[a] > e[b]) swap(a, b);
        st.push_back({a, e[a]});
        st.push_back({b, e[b]});
        e[a] += e[b]; e[b] = a;
        return true;
    }
};
