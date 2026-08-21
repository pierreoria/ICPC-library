/**
 * Author: 
 * Description: 
 */

struct segTree {
    
    int n; 
    vector<ll>nums, seg;

    segTree(int n, vector<ll>&v) : n(n), nums(v) {
        seg.resize(4*n);
        build(0, n-1, 0);
    }

    ll query(int a, int b){
        return query(a, b, 0, n-1, 0);
    }

    void update(int idx, ll num){
        update(idx, num, 0, n-1, 0);
    }

    ll query(int a, int b, int l, int u, int i){
        if(b < l || a > u) return 0;
        if(a <= l && u <= b) return seg[i];
        
        int mid = l + (u - l)/2;
        int L = 2*i + 1;
        int R = 2*i + 2;
    
        ll left = query(a, b, l, mid, L);
        ll right = query(a, b, mid+1, u, R);
        return max(left, right);
    }
    
    void update(int idx, ll num, int l, int u, int i){
        if(l == u) { seg[i] = num; return; }
    
        int mid = l + (u - l)/2;
        int L = 2*i + 1;
        int R = 2*i + 2;
        
        if(idx > mid) update(idx, num, mid+1, u, R);
        else update(idx, num, l, mid, L);
    
        seg[i] = max(seg[L], seg[R]);
    }
    
    void build(int l, int u, int i){
    
        if(l == u){ seg[i] = nums[l]; return; }
    
        int mid = l + (u - l)/2;
        int L = 2*i + 1;
        int R = 2*i + 2;
    
        build(l, mid, L);
        build(mid+1, u, R);
    
        seg[i] = max(seg[L], seg[R]); 
    
    }
};
