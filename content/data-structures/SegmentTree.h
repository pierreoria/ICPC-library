/**
 * Author: Pierre Oriá
 * Date: 2024-08-29
 * License: CC0
 * Source: AI.Cash https://codeforces.com/blog/entry/18051
 * Description: low constant factors, 2*n memory iterative seg tree
  0-based, [l,r) type queries
  seg[n] to seg[2n-1] corresponds to array[0] to array[n-1]
  for N != 2^k, cannot be used as is for efficient binary search, as there will be more than one root 
 * Time: O(\log N)
 * Status: stress-tested
 */
#pragma once
/*
  
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
