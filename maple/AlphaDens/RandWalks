RandWalks := module()
option package;
export rowsum,colsum,rownorm,colnorm,diag,specgraph,graphlap,randadj,getknn,genadj;

    rowsum := proc(C)
        m,n := Dimension(C);
        pi := Vector([seq(add(C[i,j],j=1..n),i=1..m)],datatype=float[8]);
        return pi;
    end proc;

    colsum := proc(C)
        m,n := Dimension(C);
        pi := Vector([seq(add(C[i,j],i=1..m),j=1..n)],datatype=float[8]);
        return Transpose(pi);
    end proc;

    rownorm := proc(C)
        pi := rowsum(C);
        P := DiagonalMatrix(pi,datatype=float[8])^(-1).C;
        if(nargs=2 and args[2]=true) then
            return P,pi;
        else
            return P;
        end if;
    end proc;

    colnorm := proc(C)
        pi := colsum(C);
        P := C.DiagonalMatrix(pi,datatype=float[8])^(-1);
        if(nargs=2 and args[2]=true) then
            return P,pi;
        else
            return P;
        end if;
    end proc;

#random weighted adjacency matrix
    randadj := proc(n)
        A := Matrix([seq([seq(randf(0,1),j=1..n)],i=1..n)],datatype=float[8]);
        for i from 1 to n do
            for j from i+1 to n do
            A[i,j] := A[j,i];
            end do;
        end do;
        for i from 1 to n do
            A[i,i] := 0.0;
        end do;
        return A;
    end proc;

    diag := proc(xl)
        return DiagonalMatrix(xl,datatype=float[8]);
    end proc;

#the symmetric form of the graph laplacian.
    graphlap := proc(A,rnorm)
        m := Dimension(A)[1];
        L := -A;
        for i from 1 to m do
            L[i,i] := L[i,i]-add(L[i,j],j=1..m);
        end do;
        if(nargs=2 and rnorm=true) then
            pi := rowsum(A);
            L := diag([seq(1/x,x=pi)]).L;
        end if;
        return L;
    end proc;

#laplacian operator, and spectral bases of a weighted adjacency matrix
#A, with allowable nonzero diagonal. if B1,B2 are the spectral and
#dual bases, and L is the laplacian, then we have B2.L.B1 is the
#diagonal matrix of eigenvalues. the first column of B1 is normalized
#to be all one, so that the first row of B2 is the steady state of the
#random walk.
    specgraph := proc(A)
    local x;
    uses LinAlg;
        md := module()
        option object;
        export A,P,pi,n,init,getadj,getwalk,getsteady,specbas,dualbas,geteigs,getdim,getlap,getsvd,lapmap;
        local ModulePrint;
            ModulePrint::static := proc()
                return nprintf("spectral basis of a symmetric random walk "
                               "on %d vertices",n);
            end proc;
            getadj::static := proc()
                return A;
            end proc;
            getwalk::static := proc()
                return P;
            end proc;
            getlap::static := proc(rnorm:=true)
            option remember;
                return graphlap(A,rnorm);
            end proc;
            getdim::static := proc()
                return n;
            end proc;
            getdegs::static := proc()
                return pi;
            end proc;
            getsteady::static := proc()
            option remember;
                return pi/convert(pi,`+`);
            end proc;
            getsvd::static := proc()
            option remember;
                D1 := diag([seq(1/sqrt(x),x=pi)]);
                D2 := diag([seq(sqrt(x),x=pi)]);
                C := D1.(diag(pi)+A).D1;
                B,La := SingularValues(C,output=['U','S']);
                B1,B2 := D1.B,Transpose(B).D2;
                c := B1[1,1];
                for i from 1 to n do
                    B1[i,1] := B1[i,1]/c;
                    B2[1,i] := B2[1,i]*c;
                end do;
                La := vecf([seq(max(0.0,2-x),x=La)]);
                La[1] := 0.0; #for rounding errors;
                return La,B1,B2;
            end proc;
            lapmap::static := proc(f)
                return B1.diag([seq(f(x),x=La)]).B2;
            end proc;
            geteigs::static := proc()
                return getsvd()[1];
            end proc;
            specbas::static := proc()
                return getsvd()[2];
            end proc;
            dualbas::static := proc()
                return getsvd()[3];
            end proc;
            init::static := proc()
                A := args[1];
                n := Dimension(A)[1];
                P := rownorm(A);
                pi := rowsum(A);
            end proc;
        end module;
        md:-init(A);
        return md;
    end proc;

    kermap := proc(A,t)
        spec := specgraph(A);
    end proc;

    getknn0 := proc(A::Array(datatype=float[8]),E::Array(datatype=integer[4]),R::Array(datatype=float[8]),N::integer[4],m::integer[4],k::integer[4])
        for i1 from 1 to N do
            for j from 1 to k do
                R[i1,j] := Float(infinity);
                E[i1,j] := 0;
            end do;
            for i2 from 1 to N do
                r := 0.0;
                for j from 1 to m do
                    x := A[i1,j]-A[i2,j];
                    r := r+x*x;
                end do;
                r := sqrt(r);
                if(r<R[i1,k]) then
                    for j1 from 1 to k do
                        if(r<R[i1,j1]) then
                            for j2 from k to j1+1 by -1 do
                                R[i1,j2] := R[i1,j2-1];
                                E[i1,j2] := E[i1,j2-1];
                            end do;
                            R[i1,j1] := r;
                            E[i1,j1] := i2;
                            break;
                        end if;
                    end do;
                end if;
            end do;
        end do;
    end proc;

    getknn0 := Compiler:-Compile(getknn0);

    getknn := proc(A,k)
        N,m := Dimension(A);
        E := allocla[integer[4]]([N,k]);
        R := allocla[float[8]]([N,k]);
        getknn0(A,E,R,N,m,k);
        return E;
    end proc;

    genadj := proc(typ)
        if(typ='circ') then
            return genadj0(args[2..nargs]);
        elif(typ='sphere') then
            return genadj1(args[2..nargs]);
        else
            error;
        end if;
        return ans;
    end proc;

    genadj0 := proc(n)
        A := Matrix(n,n,datatype=float[8]);
        for i from 1 to n-1 do
            A[i,i+1] := 1;
            A[i+1,i] := 1;
        end do;
        A[1,n] := 1;
        A[n,1] := 1;
        return A;
    end proc;

    genadj1 := proc(N,k)
        d := 3;
        B := matf(N,d);
        Sample(Normal(0,1),B);
        for i from 1 to N do
            r := sqrt(add(B[i,j]^2,j=1..d));
            for j from 1 to d do
                B[i,j] := B[i,j]/r;
            end do;
        end do;
        E := getknn(B,k);
        A := matf(N,N);
        for i1 from 1 to N do
            for i2 from 1 to k do
                A[i1,E[i1,i2]] := 1.0;
                A[E[i1,i2],i1] := 1.0;
            end do;
        end do;
        return A;
    end proc;

end module;
