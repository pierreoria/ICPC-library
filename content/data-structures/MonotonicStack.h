stack<int> st;

vector<int> leftSmaller(n);
vector<int> rightSmaller(n);

// Find nearest smaller element to the LEFT
for(int i = 0; i < n; i++){

    while(!st.empty() && heights[st.top()] >= heights[i]){
        st.pop();
    }

    if(st.empty())
        leftSmaller[i] = -1;
    else leftSmaller[i] = st.top();

    st.push(i);
}


// Clear the stack before scanning from the other direction
while(!st.empty()) st.pop();


// Find nearest smaller element to the RIGHT
for(int i = n - 1; i >= 0; i--){

    while(!st.empty() && heights[st.top()] >= heights[i]){
        st.pop();
    }

    if(st.empty())
        rightSmaller[i] = n;
    else rightSmaller[i] = st.top();

    st.push(i);
}
