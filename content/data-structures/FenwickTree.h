/**
 * Author: Pierre Oriá
 * Date: 2024-08-28
 * License: CC0
 * Source: codeforces/sdnr1 blog
 * Description: 1-based range sums, point increments. lower bound in O(log n)
 * Time: Both operations are $O(\log N)$.
 * Status: tested: SWERC 2023 - K
 */
#pragma once

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
