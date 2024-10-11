# read "AlphaDens.mpl";
# read "Metropolis";

# F := (x^2+y-11)^2+(x+y^2-7)^2;

# drawmap((x1,x2)->eval(F/200,[x=x1,y=x2]),[-5..5,-5..5]);

# met := Metropolis:-metmodel([F,[x,y]]);

# met:-beta := .025; A := met:-sample(1000,.1,1000); matplot(A);

# f := getkde(A,.5); mindens := cdfcut(f,.99);

# sh := densalpha(f,10000,.7,mindens);
# # check type of densalpha
# sh:-type;
# S,pow,a1 := sh:-powdata();

# drawalpha(S,pow,2.0,2);

# pow1 := rowmap(xx->eval(F,[x=xx[1],y=xx[2]])/met:-beta,A[1..300]);

# max(pow1),min(pow1);

# drawalpha(A[1..300],-pow1/5000,1,2)

# # Alpha Shape for Letter A
# Import("figures/lettera.png");

# im := %:

# Embed(im);

# A := im2mat(im);

# save A,"data/alphalet";

# A1 := sampmat(A,500); matplot(A1);

# pow1 := vecf(500);

# drawalpha(A1,pow1,.25^2,2);

# 500*542;

# read("lettera"):

# B := getbdy(A); heatmap(%);

# B1 := sampmat(B,300);

# matplot(B1);

# C := Matrix([[A1],[B1]],datatype=float[8]);

# pow2 := vecf([seq(1.0,i=1..300)]);

# pow := Vector([pow1,pow2],datatype=float[8]);

# save C,pow,"figures/alphapow";

# il := [seq(i,i=1..300),seq(i,i=501..600)]: drawalpha(10*C[il],pow[il],.51,2);

# drawalpha(10*C[il],pow[il],.1,2);

# il := [seq(i,i=1..300),seq(i,i=501..600)]: drawalpha(10*C[il],pow[il],2,2);

# drawplex:-setcolor(colormap('bioheat')(.35));

# colormap('bioheat')(.35); heatmap:-setcolor(colormap('bioheat')(.55));

# il := [seq(i,i=1..300),seq(i,i=501..600)]: drawalpha(10*C[il],pow[il],12,2);