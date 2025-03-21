read("data/mnist50"):

drawpatch0 := proc(im::Array(datatype=float[8]),A::Array(datatype=float[8]),
                   m1::integer[4],m2::integer[4],N::integer[4])
    for i0 from 1 to m1 do
        for j0 from 1 to m1 do
            c := im[i0,j0];
            for i1 from 1 to N do
                i := N*(i0-1)+i1;
                for j1 from 1 to N do
                    j := N*(j0-1)+j1;
                    A[i,j] := c;
                end do;
            end do;
        end do;
    end do;
end proc;

drawpatch0 := Compiler:-Compile(drawpatch0);

allocpatch := proc(ml)
option remember;
    return allocla[float[8]](ml);
end proc;

drawpatch := proc(im,N:=20,col:='plasma')
    ml := [arrdim(im)];
    A := allocpatch(N*ml);
    drawpatch0(im,A,op(ml),N);
    return heatmap(A,col);
end proc;

discherm0 := proc(l)
    m := 2*l+1;
    N := 2*l;
    P,aa := allocla([N,m],N);
    k := 0;
    for i from 1 to 2*l do
        k := k+1;
        P[k,i],P[k,i+1] := .5,.5;
        aa[k] := ncr(2*l-1,i-1)/2^(2*l-1);
    end do;
    return P,aa;
end proc;

ncr := proc(n,k)
    if(k<0 or k>n) then
        return 0;
    end if;
    return n!/k!/(n-k)!;
end proc;

specdiff0 := proc(P,pi)
local x;
uses LinAlg;
    N,n := Dimension(P);
    H := Transpose(P).diag(pi).P;
    pi1 := rowsum(H);
    D1 := diag([seq(1/sqrt(x),x=pi1)]);
    D2 := diag([seq(sqrt(x),x=pi1)]);
    D3 := diag([seq(1/x,x=pi1)]);
    C1 := D1.H.D1;
    U1,La1 := SingularValues(C1,output=['U','S']);
    La := Vector(map(x->sqrt(max(0.0,1-x)),La1),datatype=float[8]);
    B1,B2 := Transpose(U1).D2,D1.U1;
    L := IdentityMatrix(n,datatype=float[8])-D3.H;
    return H,B1,B2,La,L;
end proc;

discherm1 := proc(l,modes)
    if(type(modes,'integer')) then
        return discherm1(l,0..modes);
    end if;
    P,aa := discherm0(l);
    H,B1,B2,La,L := specdiff0(P,aa);
    D1 := diag(rowsum(H));
    B1 := -B2[..,[seq(i+1,i=modes)]];
    N,m := Dimension(B1);
    B2 := Matrix([seq([seq(B1[i,j]*ncr(N-1,i-1)/2^(N-1),j=1..m)],i=1..N)],datatype=float[8]);
    return B1,B2;
end proc;

discherm := proc(l,modes)
    A1,A2 := discherm1(l,max(seq(p[1],p=modes),seq(p[2],p=modes)));
    N := 2*l+1;
    m := nops(modes);
    B1,B2,V1,V2,W1,W2 := allocla([N^2,m],[N^2,m],N,N,N,N);
    k := 0;
    for p in modes do
        i1,i2 := op(p);
        getcol[V1](A1,i1+1);
        getcol[W1](A1,i2+1);
        getcol[V2](A2,i1+1);
        getcol[W2](A2,i2+1);
        k := k+1;
        j := 0;
        for j2 from 1 to N do
            for j1 from 1 to N do
                j := j+1;
                B1[j,k] := V1[j1]*W1[N-j2+1];
                B2[j,k] := V2[j1]*W2[N-j2+1];
            end do;
        end do;
    end do;
    return B1,B2;
end proc;

sinorm := proc(A,r0)
    N := Dimension(A)[1];
    ans1 := [];
    for i from 1 to N do
        r := add(A[i,j]^2,j=3..5);
        d := sqrt(r);
        if(r>r0) then
            ans1 := [op(ans1),[seq(A[i,j]/d,j=1..5)]];
        end if;
    end do;
    ans := Matrix(ans1,datatype=float[8]);
    return ans;
end proc;

sisub := proc(A,r0)
    N := Dimension(A)[1];
    ans := [];
    for i from 1 to N do
        r := add(A[i,j]^2,j=1..5);
        d := sqrt(r);
        if(d>r0) then
            ans := [op(ans),i];
        end if;
    end do;
    return snorm(A[ans]);
end proc;

snorm := proc(A,jl)
    N,m := Dimension(A);
    if(nargs=1) then
        return snorm(A,[seq(j,j=1..m)]);
    end if;
    A1 := Matrix(A,datatype=float[8]);
    for i from 1 to N do
        r := sqrt(add(A1[i,j]^2,j=jl));
        for j from 1 to m do
            A1[i,j] := A1[i,j]/r;
        end do;
    end do;
    return A1;
end proc;

snorm1 := proc(A)
    A1 := Matrix(A,datatype=float[8]);
    N := Dimension(A)[1];
    for i from 1 to N do
        r := sqrt(add(A1[i,j]^2,j=1..2));
        for j from 1 to 5 do
            A1[i,j] := A1[i,j]/r;
        end do;
    end do;
    return A1;
end proc;

snorm2 := proc(A)
    A1 := Matrix(A,datatype=float[8]);
    N := Dimension(A)[1];
    for i from 1 to N do
        r := sqrt(add(A1[i,j]^2,j=3..5));
        for j from 1 to 5 do
            A1[i,j] := A1[i,j]/r;
        end do;
    end do;
    return A1;
end proc;

mobcoords := proc(A)
    N := Dimension(A)[1];
    V := allocla(5);
    ans := [];
    for i from 1 to N do
        getrow[V](A,i);
        ans1 := [(V[3]-V[5])/sqrt(2.0),-V[4],0];
        ans2 := ans1*(V[1]/2)+[0,0,-V[2]/2];
        ans := [op(ans),ans1+1.0*ans2];
    end do;
    return Matrix(ans,datatype=float[8]);
end proc;

mobcoords1 := proc(A)
    N := Dimension(A)[1];
    V := allocla(5);
    ans := [];
    for i from 1 to N do
        getrow[V](A,i);
        ans1 := [V[1],V[2],0];
        ans2 := ans1*(V[3]-V[5])*.0+[0,0,-V[4]/2];
        ans := [op(ans),ans1+5.0*ans2];
    end do;
    return Matrix(ans,datatype=float[8]);
end proc;

drawmob0 := proc(l)
option remember;
    B1,B2 := discherm2(l,[[1,0],[0,1],[2,0],[1,1],[0,2]]);
    id := imdata(allocla([0,(2*l+1)^2]),2*l+1,2*l+1);
    return B1,id;
end proc;

drawmob := proc(V,l:=50)
local x;
    B1,id := drawmob0(l);
    A := id:-getim(B1.Transpose(V));
    a,b := min(A),max(A);
    c := min(abs(a),abs(b));
    #map[inplace](x->(c+max(min(x,c),-c))/2/c,A);
    return ImageTools:-Embed(ImageTools:-Create(A));
end proc;

patchds := proc(i,l)
    digl := digtab[i];
    ans := [];
    for k from 1 to 50 do
        arr := vec2arr(digl[k],[28,28]);
        B := vecpatches(arr,l);
        ans := [op(ans),[B]];
    end do;
    ans := matf(ans);
    return ans;
end proc;

#send to the 4-sphere, taking up to quadratic modes of the hermite
#polynomial basis
tosphere := proc(A,l,r)
    modes := [[1,0],[0,1],[2,0],[1,1],[0,2]];
    B1,B2 := discherm(l,modes);
    ans := A.B2;
    ans := sisub(ans,r);
    return ans;
end proc;
