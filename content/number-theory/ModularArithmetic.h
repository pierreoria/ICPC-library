/**
 * Author: Lukas Polacek, Joshua Andersson
 * Date: 2009-09-28
 * License: CC0
 * Source: folklore
 * Description: Operators for modular arithmetic.
 */
#pragma once

template<ll MOD> // using Mint = Mod<998244353>;
struct Mod {
    ll x;
    Mod(ll y = 0) : x((y %= MOD) < 0 ? y + MOD : y) {}
    Mod(ll y, int) : x(y) {}
    Mod operator+(Mod b) { return {x + b.x, 0}; }
    Mod operator-(Mod b) { return {x - b.x + MOD, 0}; }
    Mod operator*(Mod b) { return {x * b.x % MOD, 0}; }
    Mod operator/(Mod b) { return *this * b.inv(); }
    Mod inv() {
        ll a = x, b = MOD, u = 1, v = 0;
        while (b) {
            ll t = a / b;
            a -= t * b; swap(a, b);
            u -= t * v; swap(u, v);
        }
        assert(a == 1); return u;
    }
    Mod pow(ll e) {
        assert(e >= 0); Mod a = *this, r = 1;
        for (; e; e >>= 1, a = a * a) if (e & 1) r = r * a;
        return r;
    }
};
