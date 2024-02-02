read("include"):read "so3":
A := ImportMatrix("data/so3land1.csv",datatype=float[8]);
# Or to generate, run A := landima(1000000, 8.0); not guranteed you will get a cover

drawso3(randrow(A));

X,Phi := getalpha(A, 10.0, 4);
getbetti[true](X, 0..3, 2);

getbetti[true](X, 0..3, 3);