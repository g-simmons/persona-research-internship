LinAlg := module()
option package;
export getrow1,getrow,setrow1,setrow,getcol1,getcol,setcol1,setcol,multvv1,multvv,multmv1,multmv,multvm1,multvm,multmm1,multmm,lmult1,lmult,rmult1,rmult,invinds1,copyinds1,copyvec1,copyvec2,copyvec,copymat1,copymat2,vecsp1,vecsp2,vecsp3,vecsp,vecdist1,vecdist2,vecdist3,vecdist,allocla,appendla,resize,linsolve,shiftrows,shiftcols,getvec,vecf,veci,matf,mati,dynla,setvec,rowmap;

    getrow1 := proc(A::Array(datatype=float[8]),i::integer[4],V::Array(datatype=float[8]),n::integer[4])
        for j from 1 to n do
            V[j] := A[i,j];
        end do;
    end proc;

    getrow1 := Compiler:-Compile(getrow1);

    getrow := proc(A,i,V)
        if(type(procname,indexed)) then
            return getrow(A,i,op(procname));
        end if;
        m,n := Dimension(A);
        getrow1(A,i,V,n);
        return V;
    end proc;

    setrow1 := proc(A::Array(datatype=float[8]),i::integer[4],V::Array(datatype=float[8]),n::integer[4])
        for j from 1 to n do
            A[i,j] := V[j];
        end do;
    end proc;

    setrow1 := Compiler:-Compile(setrow1);

    setrow := proc(A,i,V)
        m,n := Dimension(A);
        setrow1(A,i,V,n);
    end proc;

    getcol1 := proc(A::Array(datatype=float[8]),j::integer[4],V::Array(datatype=float[8]),m::integer[4])
        for i from 1 to m do
            V[i] := A[i,j];
        end do;
    end proc;

    getcol1 := Compiler:-Compile(getcol1);

    getcol := proc(A,j,V)
        if(type(procname,indexed)) then
            return getcol(A,j,op(procname));
        end if;
        m,n := Dimension(A);
        getcol1(A,j,V,m);
        return V;
    end proc;

    setcol1 := proc(A::Array(datatype=float[8]),j::integer[4],V::Array(datatype=float[8]),m::integer[4])
        for i from 1 to m do
            A[i,j] := V[i];
        end do;
    end proc;

    setcol1 := Compiler:-Compile(setcol1);

    setcol := proc(A,j,V)
        m,n := Dimension(A);
        setcol1(A,j,V,m);
    end proc;

    multvv1 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),W::Array(datatype=float[8]),n::integer[4])
        for i from 1 to n do
            W[i] := U[i]*V[i];
        end do;
        return;
    end proc;

    multvv1 := Compiler:-Compile(multvv1);

    multvv := proc(U,V,W)
        multvv1(U,V,W,Dimension(U));
    end proc;

    multmv1 := proc(A::Array(datatype=float[8]),U::Array(datatype=float[8]),V::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for i from 1 to m do
            c := 0.0;
            for j from 1 to n do
                c := c+A[i,j]*U[j];
            end do;
            V[i] := c;
        end do;
        return;
    end proc;

    multmv1 := Compiler:-Compile(multmv1);

    multmv := proc(A,U,V)
        multmv1(A,U,V,Dimension(A));
    end proc;

    multvm1 := proc(U::Array(datatype=float[8]),A::Array(datatype=float[8]),V::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for j from 1 to n do
            c := 0.0;
            for i from 1 to m do
                c := c+U[i]*A[i,j];
            end do;
            V[j] := c;
        end do;
        return;
    end proc;

    multvm1 := Compiler:-Compile(multvm1);

    multvm := proc(U,A,V)
        multvm1(U,A,V,Dimension(A));
    end proc;

    multmm1 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),C::Array(datatype=float[8]),l::integer[4],m::integer[4],n::integer[4])
        for i from 1 to l do
            for j from 1 to n do
                c := 0.0;
                for k from 1 to m do
                    c := c+A[i,k]*B[k,j];
                end do;
                C[i,j] := c;
            end do;
        end do;
        return;
    end proc;

    multmm1 := Compiler:-Compile(multmm1);

    multmm := proc(A,B,C)
        multmm1(A,B,C,Dimension(B),Dimension(C)[2]);
    end proc;

#inplace multiplication
    lmult1 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),m::integer[4],n::integer[4],V::Array(datatype=float[8]))
        for k from 1 to n do
            for j from 1 to m do
                c := 0.0;
                for i from 1 to m do
                    c := c+A[j,i]*B[i,k];
                end do;
                V[j] := c;
            end do;
            for j from 1 to m do
                B[j,k] := V[j];
            end do;
        end do;
    end proc;

    lmult1 := Compiler:-Compile(lmult1);

    lmult := proc(A,B)
        m,n := Dimension(B);
        if(nargs=3) then
            V := args[3];
        else
            V := allocla[float[8]](m);
        end if;
        lmult1(A,B,m,n,V);
        return B;
    end proc;

#inplace multiplication
    rmult1 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),m::integer[4],n::integer[4],V::Array(datatype=float[8]))
        for i from 1 to m do
            for j from 1 to n do
                c := 0.0;
                for k from 1 to n do
                    c := c+A[i,k]*B[k,j];
                end do;
                V[j] := c;
            end do;
            for j from 1 to n do
                A[i,j] := V[j];
            end do;
        end do;
    end proc;

    rmult1 := Compiler:-Compile(rmult1);

    rmult := proc(A,B)
        m,n := Dimension(A);
        if(nargs=3) then
            V := args[3];
        else
            V := allocla[float[8]](n);
        end if;
        rmult1(A,B,m,n,V);
        return A;
    end proc;

    copyinds1 := proc(V::Array(datatype=integer[4]),W::Array(datatype=float[8]),N::integer[4])
        for i from 1 to N do
            W[i] := V[i];
        end do;
    end proc;

    copyinds1 := Compiler:-Compile(copyinds1);

    invinds1 := proc(inds1::Array(datatype=integer[4]),inds2::Array(datatype=integer[4]),n1::integer[4],N::integer[4])
        for k from 1 to N do
            inds2[k] := 0;
        end do;
        for k from 1 to n1 do
            j := inds1[k];
            inds2[j] := k;
        end do;
    end proc;

    invinds1 := Compiler:-Compile(invinds1);

    copyvec1 := proc(V::Array(datatype=float[8]),W::Array(datatype=float[8]),N::integer[4])
        for i from 1 to N do
            W[i] := V[i];
        end do;
    end proc;

    copyvec1 := Compiler:-Compile(copyvec1);

    copyvec2 := proc(V::Array(datatype=float[8]),J::Array(datatype=integer[4]),W::Array(datatype=float[8]),n::integer[4])
        for i from 1 to n do
            W[i] := V[J[i]];
        end do;
    end proc;

    copyvec2 := Compiler:-Compile(copyvec2);

    copyvec := proc(U,V)
        if(not type(U,'Vector')) then
            for i from 1 to nops(U) do
                V[i] := U[i];
            end do;
        end if;
        if(nargs=2) then
            copyvec1(args,Dimension(U));
        elif(nargs=3) then
            copyvec2(args,Dimension(args[2]));
        end if;
    end proc;

    copymat1 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for i from 1 to m do
            for j from 1 to n do
                B[i,j] := A[i,j];
            end do;
        end do;
    end proc;

    copymat1 := Compiler:-Compile(copymat1);

    copymat2 := proc(A::Array(datatype=float[8]),J::Array(datatype=integer[4]),B::Array(datatype=float[8]),n::integer[4],m::integer[4])
        for i from 1 to n do
            i1 := J[i];
            for j from 1 to m do
                B[i,j] := A[i1,j];
            end do;
        end do;
    end proc;

    copymat2 := Compiler:-Compile(copymat2);

    resize0 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),N::integer[4],m::integer[4])
        for i from 1 to N do
            for j from 1 to m do
                B[i,j] := A[i,j];
            end do;
        end do
    end proc;

    resize0 := Compiler:-Compile(resize0);

    resize1 := proc(A::Array(datatype=integer[4]),B::Array(datatype=integer[4]),N::integer[4],m::integer[4])
        for i from 1 to N do
            for j from 1 to m do
                B[i,j] := A[i,j];
            end do;
        end do
    end proc;

    resize1 := Compiler:-Compile(resize1);

    resize := proc(A,N1)
        if(not type(procname,indexed)) then
            return resize[float[8]](A,N1);
        end if;
        typ := op(procname);
        N,m := Dimension(A);
        ans := Matrix(N1,m,datatype=typ);
        copymat1(A,ans,N,m);
        return ans;
    end proc;

    allocla := proc()
        if(not type(procname,indexed)) then
            return allocla[float[8]](args);
        end if;
        typ := op(procname);
        ans := [];
        for x in args do
            if(type(x,'integer')) then
                A := Vector(x,datatype=typ);
            elif(type(x,'list')) then
                A := Matrix(op(x),datatype=typ);
            elif(type(x,'Matrix')) then
                A := Matrix(x,datatype=typ);
            elif(type(x,'Vector')) then
                A := Vector(x,datatype=typ);
            else
                error;
            end if;
            ans := [op(ans),A];
        end do;
        return op(ans);
    end proc;

    appendla := proc()
        if(type(args[1],'list')) then
            return seq(appendla(op(x)),x=args);
        end if;
        if(type(args[1],'Matrix')) then
            return Matrix([seq([A],A=args)]);
        elif(type(args[1],'Vector')) then
            return Vector([seq(V,V=args)]);
        end if;
    end proc;

    linsolve := proc(A,V)
        m,n := Dimension(A);
        sol := LinearSolve(A,V);
        tl := [op(indets(sol))];
        k := nops(tl);
        U := Vector(eval(sol,[seq(t=0,t=tl)]),datatype=float[8]);
        B := Matrix([seq([seq(coeff(sol[i],tl[j]),j=1..k)],i=1..n)],datatype=float[8]);
        return U,B;
    end proc;

    vecsp1 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),n::integer[4])
        a := 0.0;
        for i from 1 to n do
            a := a+U[i]*V[i];
        end do;
        return a;
    end proc;

    vecsp1 := Compiler:-Compile(vecsp1);

    vecsp2 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),G::Array(datatype=float[8]),n::integer[4])
        a := 0.0;
        for i from 1 to n do
            b := 0.0;
            for j from 1 to n do
                b := b+G[i,j]*V[j];
            end do;
            a := a+U[i]*b;
        end do;
        return a;
    end proc;

    vecsp2 := Compiler:-Compile(vecsp2);

    vecsp3 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),G::Array(datatype=float[8]),n::integer[4])
        a := 0.0;
        for i from 1 to n do
            a := a+G[i]*U[i]*V[i];
        end do;
        return a;
    end proc;

    vecsp3 := Compiler:-Compile(vecsp3);

    vecsp := proc(U,V,G)
        if(type(procname,indexed)) then
            return vecsp(U,V,op(procname));
        end if;
        n := Dimension(U);
        if(nargs=2) then
            return vecsp1(U,V,n);
        elif(type(G,'Matrix')) then
            return vecsp2(U,V,G,n);
        elif(type(G,'Vector')) then
            return vecsp3(U,V,G,n);
        end if;
    end proc;

    vecdist1 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),n::integer[4])
        a := 0.0;
        for i from 1 to n do
            b := V[i]-U[i];
            a := a+b*b;
        end do;
        a := sqrt(a);
        return a;
    end proc;

    vecdist1 := Compiler:-Compile(vecdist1);

    vecdist2 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),G::Array(datatype=float[8]),n::integer[4])
        a := 0.0;
        for i from 1 to n do
            b := 0.0;
            for j from 1 to n do
                b := b+G[i,j]*(V[j]-U[j]);
            end do;
            a := a+(V[i]-U[i])*b;
        end do;
        a := sqrt(a);
        return a;
    end proc;

    vecdist2 := Compiler:-Compile(vecdist2);

    vecdist3 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),G::Array(datatype=float[8]),n::integer[4])
        a := 0.0;
        for i from 1 to n do
            b := U[i]-V[i];
            a := a+G[i]*b*b;
        end do;
        a := sqrt(a);
        return a;
    end proc;

    vecdist3 := Compiler:-Compile(vecdist3);

    vecdist := proc(U,V,G)
        if(type(procname,indexed)) then
            return vecdist(U,V,op(procname));
        end if;
        n := Dimension(U);
        if(nargs=2) then
            return vecdist1(U,V,n);
        elif(type(G,'Matrix')) then
            return vecdist2(U,V,G,n);
        elif(type(G,'Vector')) then
            return vecdist3(U,V,G,n);
        end if;
    end proc;

    shiftrows1 := proc(A::Array(datatype=float[8]),V::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for j from 1 to n do
            c := V[j];
            for i from 1 to m do
                A[i,j] := A[i,j]+c;
            end do;
        end do;
    end proc;

    shiftrows1 := Compiler:-Compile(shiftrows1);

    shiftrows := proc(A,V)
        shiftrows1(A,V,Dimension(A));
    end proc;

    shiftcols1 := proc(A::Array(datatype=float[8]),V::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for i from 1 to m do
            c := V[i];
            for j from 1 to n do
                A[i,j] := A[i,j]+c;
            end do;
        end do;
    end proc;

    shiftcols1 := Compiler:-Compile(shiftcols1);

    shiftcols := proc(A,V)
        shiftcols1(A,V,Dimension(A));
    end proc;

    setvec := proc(V,x)
        for j from 1 to numelems(x) do
            V[j] := x[j];
        end do;
        return;
    end proc;

    getvec := proc(xl)
        if(not type(procname,indexed)) then
            return Vector(xl);
        end if;
        typ := op(procname);
        return Vector(xl,datatype=typ);
    end proc;

    vecf := proc(xl)
        return Vector(args,datatype=float[8]);
    end proc;

    veci := proc(xl)
        return Vector(args,datatype=integer[4]);
    end proc;

    matf := proc(A)
        return Matrix(args,datatype=float[8]);
    end proc;

    mati := proc(A)
        return Matrix(args,datatype=integer[4]);
    end proc;

#collection of vectors, matrices which can be resized
    dynla := proc()
        md := module()
        option object;
        export N,allocif,init,getelts,elts,typs,l;
        local ModulePrint;
            ModulePrint::static := proc()
                return nprintf("dynamic storage. %d objects, size=%d",3,N);
            end proc;
            allocif::static := proc(n)
                if(n<=N) then
                    return false;
                end if;
                N1 := 2^ceil(log(n)/log(2));
                elts1 := [];
                for i from 1 to l do
                    elt := elts[i];
                    typ := typs[i];
                    if(type(elt,'Matrix')) then
                        N0,m := Dimension(elt);
                        elt1 := Matrix(N1,m,datatype=typ);
                        elt1[1..N0,1..m] := elt;
                    elif(type(elt,'Vector')) then
                        N0 := Dimension(elt);
                        elt1 := Vector(N1,datatype=typ);
                        elt1[1..N0] := elt;
                    end if;
                    elts1 := [op(elts1),elt1];
                end do;
                elts := elts1;
                N := N1;
                return true;
            end proc;
            getelts::static := proc()
                return op(elts);
            end proc;
            init::static := proc()
                N := 16;
                elts := [];
                typs := [];
                for x in args do
                    if(type(x,'function')) then
                        typ := op(0,x);
                        m := op(x);
                        elt := Matrix(N,m,datatype=typ);
                    else
                        typ := x;
                        elt := Vector(N,datatype=typ);
                    end if;
                    elts := [op(elts),elt];
                    typs := [op(typs),typ];
                end do;
                l := nops(elts);
            end proc;
        end module;
        md:-init(args);
        return md;
    end proc;

    rowmap := proc(f,A)
        if(type(A,'numeric')) then
            N := A;
            m := numelems(f());
            V := allocla[float[8]](m);
            ans := matf(N,m);
            for i from 1 to N do
                V1 := f();
                for j from 1 to m do
                    ans[i,j] := V1[j];
                end do;
            end do;
            return ans;
        end if;
        if(nargs=3) then
            N := args[3];
        else
            N := Dimension(A)[1];
        end if;
        if(N=0) then
            error "no elements";
        end if;
        m := Dimension(A)[2];
        V := allocla[float[8]](m);
        getrow(A,1,V,m);
        V1 := f(V);
        if(type(V1,'numeric')) then
            ans := allocla[float[8]](N);
            for i from 1 to N do
                getrow1(A,i,V,m);
                ans[i] := f(V);
            end do;
        elif(type(V1,'Vector') or type(V1,'list')) then
            m1 := numelems(V1);
            ans := allocla[float[8]]([N,m1]);
            for i from 1 to N do
                getrow1(A,i,V,m);
                U := f(V);
                for j from 1 to m1 do
                    ans[i,j] := U[j];
                end do;
            end do;
        else
            error;
        end if;
        return ans;
    end proc;

end module;