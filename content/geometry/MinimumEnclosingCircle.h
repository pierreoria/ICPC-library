/**
 * Description: Computes the minimum circle that encloses a set of points.
 * Time: expected O(n)
 * Status: stress-tested
 */
#pragma once
double ccRadius(P& A, P& B, P& C) {
  return len(B, A)*len(C,B)*len(A,C)/
      abs(cross((B-A), (C-A)))/2.0;
}
 
P ccCenter(P& A, P& B, P& C) {
  P b = C-A, c = B-A;
  return A + perp(b*dist2(c)-c*dist2(b))/cross(b, c)/2;
}
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
pair<P, double> mec(vector<P>& pts){
	shuffle(begin(pts),end(pts),rng);
	P o = pts[0];
	const double EPSS = 1+1e-8;
	double r = 0;
	for(int i = 0; i < pts.size(); i++) if(len(o, pts[i]) > r * EPSS){
		o = pts[i], r = 0;
		for(int j = 0; j < i; j++) if(len(o, pts[j]) > r * EPSS){
			o = (pts[i]+pts[j])/2.0;
			r = len(o, pts[i]);
			for(int k = 0; k < j; k++) if(len(o, pts[k]) > r * EPSS){
				o = ccCenter(pts[i],pts[j],pts[k]);
				r = len(o, pts[i]);
			}
		}
	}
	return {o, r};
}
