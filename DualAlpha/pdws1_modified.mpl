read("include"):
pd := randpd(100);
A := ImportMatrix("data/minimizers_sample.csv");
[m, n] := Dimension(A);
A := A[2.., ..];
A := Matrix(A,datatype=float[8]);
powervec := A[..,3];
A := A[..,..2];
A := Matrix(A,datatype=floar[8]);
pd := pdiag(A, powervec, 3);

pd:-draw();

X,Phi := getalpha(pd:-A,pd:-pow,pd:-a1,3);
X,Phi := getalpha(pd:-A,pd:-pow,3,3);

getbetti(x,0..2,3);

display([pd:-draw(),seq(point(Phi(sig),color=blue,symbol=solidcircle,symbolsize=5),sig=X[0]),seq(point(Phi(sig),color=red,symbol=solidcircle,symbolsize=5),sig=X[1]),seq(point(Phi(sig),color=red,symbol=solidcircle,symbolsize=5),sig=X[2])]);