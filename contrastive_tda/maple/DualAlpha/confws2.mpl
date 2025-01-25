read("include"):
read("../SF") : withSF(); read("config"):

A := ImportMatrix("data/confland.csv",datatype=float[8]);
run A := confpd4(1000000,.3);

X,Phi := getalpha(A,.35,7);

frobconf(X,[2,2],3,5);
