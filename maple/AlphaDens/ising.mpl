specis := proc(E)
    n := Dimension(E)[1];
    E1 := Matrix(E,datatype=float[8]);
    V1 := rowsum(E1);
    for i from 1 to n do
        E1[i,i] := V1[i]-E1[i,i];
    end do;
    V2 := rowsum(E1);
    D1 := DiagonalMatrix([seq(1/sqrt(x),x=V2)],datatype=float[8],shape=diagonal);
    D2 := DiagonalMatrix([seq(sqrt(x),x=V2)],datatype=float[8],shape=diagonal);
    D3 := DiagonalMatrix([seq(1/x,x=V2)],datatype=float[8],shape=diagonal);
    C1 := D1.E1.D1;
    U1,La := SingularValues(C1,output=['U','S']);
    La := Vector(map(x->sqrt(max(0.0,1-x)),La),datatype=float[8]);
    B1,B2 := D2.U1,Transpose(U1).D1;
    L1 := IdentityMatrix(n,datatype=float[8])-E1.D3;
    return B1,B2,L1,La;
end proc;

specdiff0 := proc(P,pi)
local x;
uses LinAlg;
    print(P,pi);
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

stochis1 := proc(E)
    N := round(convert(E,`+`));
    n := Dimension(E)[1];
    k := 0;
    P := allocla([N,n]);
    for i from 1 to n do
        for j from i+1 to n do
            if(E[i,j]=1) then
                k := k+1;
                P[k,i],P[k,j] := .5,.5;
            end if;
        end do;
    end do;
    aa := Vector([seq(1/N,i=1..N)],datatype=float[8]);
    return P,aa;
end proc;

specdiff := proc(P,aa)
    argl := [args];
    md := module()
    option object;
    export P,H,N,n,B1,B2,La,L,aa,bb,Q,getspec,tospec,tofunc,res,blend,proj,sp;
    local init,ModulePrint;
        ModulePrint::static := proc()
            return nprintf("spectral basis R^%d->R^%d",N,n);
        end proc;
        res::static := proc(U)
            return P.U;
        end proc;
        lift::static := proc(V)
            return Q.V;
        end proc;
        getspec::static := proc(i)
            return B2[..,i];
        end proc;
        tospec::static := proc(U)
            return B1.U;
        end proc;
        tofunc::static := proc(V)
            return B2.V;
        end proc;
        blend::static := proc(V,t)
            return diag([seq(exp(-t*La[i]^2),i=1..n)]).V;
        end proc;
        sp::static := proc(V1,V2)
            return add(bb[i]*V1[i]*V2[i],i=1..n);
        end proc;
        proj::static := proc(V,il)
            V1 := allocla(V);
            ArrayTools:-Fill(0.0,V1);
            for i in il do
                V1[i] := V[i];
            end do;
            return V1;
        end proc;
        init::static := proc()
        local x;
            P,aa := args;
            N,n := Dimension(P);
            H,B1,B2,La,L := specdiff0(P,aa);
            bb := rowsum(H);
            D1 := diag(map(x->1/x,bb));
            P1 := D1.H;
            Q,bb1 := bayesmap(P,aa);
        end proc;
        init(op(argl));
    end module;
    return md;
end proc;

bayesmap := proc(P,bb)
local p;
    N1,N := Dimension(P);
    aa := allocla(N);
    multvm[aa](bb,P);
    La1 := DiagonalMatrix(map(x->1/x,aa),shape=diagonal,datatype=float[8]);
    La2 := DiagonalMatrix(bb,shape=diagonal,datatype=float[8]);
    ans := La1.Transpose(P).La2;
    return ans,aa;
end proc;

