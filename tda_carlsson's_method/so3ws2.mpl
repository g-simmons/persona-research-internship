read("include"):
read "so3":
A := ImportMatrix("data/so3land2.csv",datatype=float[8]);
# or generate them using A := landimc(1000000, 3.0); no guarntee they will cover

drawso3['HSV'](randelt(A));

X,Phi := getalpha(A, 3.5, 4);
getbetti[true](X, 0..3, 3);
getbetti[true](X, 0..3, 2);

