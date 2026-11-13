/**
 * Author: 
 * Description: 
 */

const int N = 1e6 + 5;
ll mob[N]; bool prime[N];
void sieve(){
    for(int i = 0; i < N; i++){
        mob[i] = 1;
        prime[i] = 1;
    }

    mob[1] = 1;
    prime[0] = prime[1] = 0;
    for(int i = 2; i < N; i++){
        if(prime[i]){
            for(int j = i; j < N; j += i){
                if(j > i) prime[j] = 0;
                if((j / i) % i == 0) mob[j] = 0;
                mob[j] *= -1;
            }
        }
    }
}
