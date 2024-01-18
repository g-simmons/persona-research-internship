read("include"):
read("../SF") : withSF();

read("config"):
A := confpd3(1000000,03);

drawconf(A[1]); drawconf(A[2]); drawconf(A[3]); drawconf(A[4]); drawconf(A[5]); drawconf(A[6]); 

X,Phi := getalpha(A,.35,4);
getbetti(X,0..3,2);
getbetti(X,0..3,2);

frobconf(X,3,3,5);
sig := randelt(X[2]); display([drawconf(Phi(sig)),seq(drawconf(Phi([i])),i=sig)]);
