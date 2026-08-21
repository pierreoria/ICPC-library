/**
 * Author: 
 * Description: 
 */

const int MAXV = 1e6 + 5;
ll mob[MAXV]; 
bool isprime[MAXV];

void sieve(){
    
    for(int i = 0; i < MAXV; i++){
        mob[i] = 1;
        isprime[i] = 1;
    }

    isprime[0] = isprime[1] = 0;
    mob[1] = 1;

    for(int i = 2; i<MAXV; i++){
        if(isprime[i]){
            for(int j = i; j<MAXV; j+=i){
                if(j > i) isprime[j] = 0;
                mob[j] *= -1;
                if((j/i) % i == 0) mob[j] = 0;
            }
        }
    }
}
