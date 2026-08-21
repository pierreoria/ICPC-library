void solve(){
    int n; cin >> n;
    vector<int>v(n);
    for(auto &x: v) cin >> x;
    vector<int>sz(n+1, 0);
    int i = 0, j = 0;
    while(i < n){
        int aux = v[i]; 
        j = i;
        while(j < n && v[j] <= aux) j++;
        sz[j-i]++;
        i = j;
    }   
    
    for(int i = 1; i<=n; i++){
        if(sz[i]){
            int rem = (sz[i] - 1)/2;
            if(rem){
                sz[i] -= 2* rem;
                sz[2*i] += rem;
            }
        }
    }

    vector<int>sz2;
    for(int i = 1; i<=n; i++){
        for(int j = 0; j<sz[i]; j++){
            sz2.push_back(i);
        }
    }

    bitset<250001>b;
    b[0] = 1;
    for(int x: sz2){
        b |= b << x;
    }

    cout << (b[n/2] ? "Yes" : "No") << endll;
}
