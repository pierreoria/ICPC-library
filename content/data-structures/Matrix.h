const int MOD=1e9+7;
const int ms =2;
 
class Matrix{
 public:
   ll mat[ms][ms] = {{0,0},{0,0}};
   //vector<vector<ll>> mat= vector<vector<ll>>(n,vector<ll>(n));
 
   Matrix operator * (const Matrix &p){
 
     Matrix ans;
     for(int i = 0; i<ms; i++){ 
       for(int j = 0; j<ms; j++){
         for(int k = ans.mat[i][j] = 0; k<ms;k++){
         
           ans.mat[i][j] = ((ans.mat[i][j] + 1LL * (mat[i][k]%MOD) *(p.mat[k][j]%MOD)))%MOD;
 
         }
       }
     }
     return ans;
 
   }
};
 
Matrix mfxp(Matrix a, ll b){
 
 Matrix ans;
 for(int i=0;i<ms;i++){
   ans.mat[i][i] = 1;
 }
 
 while(b){
   if(b&1) ans = ans*a;
   a = a*a;
   b>>=1; 
 }
 
 return ans;
 
}
