/**
 * Author: Simon Lindholm
 * Date: 2016-10-08
 * License: CC0
 * Source: me
 * Description: Range assignment, point and range queries segment tree. 
 * 0-based, closed intervals. root at index 1, responsible for array[0:n-1]
 * Time: O(\log N).
 * Status: tested on 2024 ICPC Bolivia pre-national - G. Must be changed to fit specific problem
 */
#pragma once


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
