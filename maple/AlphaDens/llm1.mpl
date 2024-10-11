Import("data/llm_edited_reviews_embedded_with_sentiment.csv");

A0 := convert(%,'Matrix'): A := Matrix(%,datatype=float[8]);

A := Matrix(A0[..,1..1023],datatype=float[8]);

sh := densalpha(A,2.0,10000,.8,.9);

sh:-drawplex(2,.8);

R := rowmap(x->sqrt(add(x[i]^2,i=1..40)),A1);

A2 := matf([seq([seq(A[i,j]/R[i],j=1..1023)],i=1..270)]);

add(A1[1,j]^2,j=1..1023);

sh := densalpha(A1,.05,10000,.8,.9);

sh:-drawplex(2,.8);

A2 := pca(A)[..,1..10];

sh := densalpha(A3,.3,10000,.5,.9);

X := sh:-getplex(2,.5);

drawplex(X,sh:-S[..,1..3].randframe(3,2));

matplot(A3[..],sh:-S);

A[1..10,1];

A[1..10,2];

max(A[..,2]);

B1 := Transpose(A)[..,1..2];

B1[1..10];

A[1..5,1..5];

A;

A := A[2..271];

A3 := pca(A2)[..,1..40];

max(A);

max(A3);

for i from 1 to 270 do A3[i,1] := A3[i,1]/3; end do;