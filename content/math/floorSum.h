/**
 * Author: Simon Lindholm
 * Date: 2018-07-06
 * License: CC0
 * Description: //todo
 * Time: O(log(a))
 */
#pragma once

long long floorSum(long long a, long long b, long long c, long long n){
    if(!a) return (b / c) * (n + 1);
    if(a >= c or b >= c) {
        return ((n * (n + 1)) / 2) * (a / c) + (n + 1)
                * (b / c) + floorSum(a % c, b % c, c, n);
    }
    long long m = (a * n + b) / c;
    return m * n - floorSum(c, c - b - 1, a, m - 1);
}