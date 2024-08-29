/**
 * Author: Pierre Oriá
 * Date: 2024-08-28
 * License: CC0
 * Source: cp algorithms
 * Description: vanilla range update seg, sum point queries
 0-based
 root = index 1 (tl = 0, tr = n-1)
 closed intervals -> [l,r] type queries
 * Time: Both operations are $O(\log N)$.
 * Status: --
 */
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
