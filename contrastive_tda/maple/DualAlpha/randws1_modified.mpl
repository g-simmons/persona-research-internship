read("include"):
A := Matrix([seq([seq(randf(-1,1),j=1..2)],i=1..100)],datatype=float[8]);

X,W := getalpha(A,1.0,3);

drawalpha2(X,W)

