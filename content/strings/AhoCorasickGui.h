const int ALPHA = 26, off = 'a';
struct Node {
    int p, sl, ol;
    int idw;
    bool vis = false;
    array<int, ALPHA> nxt;

    Node(){ nxt.fill(0); sl = 0; p = 0; ol = 0; idw = -1;}
};
typedef Node* trie;
struct Aho {
    vector<Node>nodes;
    vector<int>eq;
    int nwords = 0;

    Aho(){
        nodes.emplace_back();
    }

    void add(string &s){
        int t = 0;
        for(auto c : s){ c -= off;
            if(!nodes[t].nxt[c]){
                nodes[t].nxt[c] = nodes.size();
                nodes.emplace_back();
                nodes.back().p = t;
            }
            t = nodes[t].nxt[c];
        }
        nodes[t].idw = nwords++;
        eq.emplace_back(t);
    }

    void buildSufixLink(){
        queue<int>q;

        for(int c = 0; c<ALPHA; c++) 
            if(nodes[0].nxt[c]) 
                q.push(nodes[0].nxt[c]);
        

        while(!q.empty()){
            int u = q.front(); q.pop();

            for(int c = 0; c<ALPHA; c++){
                int v = nodes[u].nxt[c];
                if(v){

                    nodes[v].sl = nodes[nodes[u].sl].nxt[c];
                    
                    int fail = nodes[v].sl;
                    nodes[v].ol = (nodes[fail].idw != -1) ? fail : nodes[fail].ol;
                    
                    q.push(v);

                } else nodes[u].nxt[c] = nodes[nodes[u].sl].nxt[c];
            }
        }
    }

    void findPattern(string &s){
        int u = 0, sz = s.size();
        for(int i = 0; i<sz; i++){
            int c = s[i] - off;
            u = nodes[u].nxt[c];

            int aux = (nodes[u].idw == -1) ? nodes[u].ol : u;

            while(aux && !nodes[aux].vis){
                nodes[aux].vis = 1;
                aux = nodes[aux].ol;
            }
        }
    }

};
