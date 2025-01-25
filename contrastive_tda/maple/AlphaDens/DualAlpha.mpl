DualAlpha := module()
option package;
export pdiag,randpd,getalpha,alphaplex,drawpd,pcell,pgraph,pcech,witmap,vectab,drawalpha,alphastep,powheat;
local drawpow,drawsv,metpair;

    pcell0 := proc(A::Array(datatype=float[8]),V::Array(datatype=float[8]),pow::Array(datatype=float[8]),p0::float[8],B::Array(datatype=float[8]),U::Array(datatype=float[8]),n,m)
        for i1 from 1 to n do
            for i2 from 1 to n do
                c := 0.0;
                for j from 1 to m do
                    c := c+(A[i1,j]-V[j])*(A[i2,j]-V[j]);
                end do;
                B[i1,i2] := c;
            end do;
        end do;
        for i from 1 to n do
            U[i] := -(p0-pow[i]+B[i,i])/2;
        end do;
        return;
    end proc;

    pcell0 := Compiler:-Compile(pcell0);

#load the active set.
    pcell1 := proc(A::Array(datatype=float[8]),V::Array(datatype=float[8]),ll::Array(datatype=float[8]),active::Array(datatype=integer[4]),wit::Array(datatype=float[8]),n::integer[4],m::integer[4])
        for j from 1 to m do
            wit[j] := V[j];
        end do;
        for i from 1 to n do
            i1 := active[i];
            if(i1=0) then
                break;
            end if;
            c := ll[i];
            for j from 1 to m do
                wit[j] := wit[j]-c*(A[i1,j]-V[j]);
            end do;
        end do;
    end proc;

    pcell1 := Compiler:-Compile(pcell1);

#backend object for a cell in the power diagram cell, based on dual
#programming. the with powers are given by A,pow, with center object
#assumed at the origin, with power p0. loadseqs loads the equations
#for the dual programming, while setface returns the filtered value
#of a given face. if not greater than b, the representative is stored
#in wit.
    pcell := proc(A,pow,p0)
        md := module()
        option object;
        export A,pow,V,p0,n1,n,m,B,U,l0,wit,a,setcent,setcent1,setcent2,setsites,setsites1,setsites2,loadeqs,setface,primtol,singtol,active,alloc,ll,init;
        local ModulePrint;
            ModulePrint::static := proc()
                s := "dual power cell, %d vertices in R^%d";
                return nprintf(s,n,m);
            end proc;
            setcent1::static := proc(V1,p1)
                copyvec1(V1,V,m);
                p0 := p1;
                return;
            end proc;
            setcent::static := setcent1;
            setcent2::static := proc(A0,pow0,i)
                getrow1(A0,i,V,m);
                p0 := pow0[i];
            end proc;
            setsites::static := proc(A0,pow0)
                setsites1(A0,pow0,Dimension(A0)[1]);
            end proc;
            setsites1::static := proc(A0,pow0,n0)
                n := n0;
                copymat1(A0,A,n,m);
                copyvec1(pow0,pow,n);
                return;
            end proc;
            setsites2::static := proc(A0,pow0,J,n0)
                n := n0;
                copymat2(A0,J,A,n,m);
                copyvec2(pow0,J,pow,n);
                return;
            end proc;
            loadeqs::static := proc()
                pcell0(A,V,pow,p0,B,U,n,m);
                return;
            end proc;
            setface::static := proc(sig,b)
                ArrayTools:-Fill(0,active);
                l := nops(sig);
                for i from 1 to l do
                    active[i] := sig[i];
                end do;
                objmax := (b+p0)/2;
                r := dualalg(B,U,n,active,l,n,objmax,primtol,singtol,op(alloc));
                if(r>=objmax-primtol) then
                    return false;
                end if;
                pcell1(A,V,ll,active,wit,n,m);
                a := 2*r-p0;
                return true;
            end proc;
            init::static := proc()
                A,pow,p0 := args;
                n,m := Dimension(A);
                B,U,V,wit := allocla[float[8]]([n,n],n,m,m);
                active := allocla[integer[4]](n);
                primtol := .000001;
                singtol := .00000001;
                l := 0;
                alloc := dualalgalloc(n,n);
                ll := alloc[1];
                n1 := n;
            end proc;
        end module;
        if(type(args[1],'Matrix')) then
            md:-init(A,pow,p0);
            md:-seteqs();
        elif(type(args[1],'numeric')) then
            n,m := args;
            md:-init(allocla([n,m],n),0.0);
        end if;
        return md;
    end proc;

    #compute the maximum degree of any vertex in the graph which is the
#nerve of the corresponding weighted ball covering, for
#preprocessing.
    pcech0 := proc(A::Array(datatype=float[8]),pow::Array(datatype=float[8]),a1::float[8],N::integer[4],m::integer[4])
        N1 := 0;
        for i1 from 1 to N do
            s1 := pow[i1]+a1;
            if(s1<0.0) then
                next;
            end if;
            s1 := sqrt(s1);
            for i2 from i1+1 to N do
                s2 := pow[i2]+a1;
                if(s2<0.0) then
                    next;
                end if;
                s2 := sqrt(s2);
                s := s1+s2;
                s := s*s;
                r := 0.0;
                for j from 1 to m do
                    c := A[i1,j]-A[i2,j];
                    r := r+c*c;
                    if(r>s) then
                        break;
                    end if;
                end do;
                if(r<=s) then
                    N1 := N1+1;
                end if;
            end do;
        end do;
        return N1;
    end proc;

    pcech0 := Compiler:-Compile(pcech0);

#compute the above intersection graph by storing the neighbors to
#every vertex into E.
    pcech1 := proc(A::Array(datatype=float[8]),pow::Array(datatype=float[8]),a1::float[8],E::Array(datatype=integer[4]),R::Array(datatype=float[8]),W::Array(datatype=float[8]),N::integer[4],m::integer[4])
        N1 := 0;
        for i1 from 1 to N do
            p1 := pow[i1];
            s1 := p1+a1;
            if(s1<0.0) then
                next;
            end if;
            s1 := sqrt(s1);
            k := 0;
            for i2 from 1 to N do
                if(i2=i1) then
                    next;
                end if;
                p2 := pow[i2];
                s2 := p2+a1;
                if(s2<0.0) then
                    next;
                end if;
                s2 := sqrt(s2);
                s := s1+s2;
                s := s*s;
                r := 0.0;
                b := 0.0;
                for j from 1 to m do
                    c := A[i1,j]-A[i2,j];
                    r := r+c*c;
                    if(r>s) then
                        break;
                    end if;
                end do;
                if(r<=s) then
                    N1 := N1+1;
                    E[N1,1] := i1;
                    E[N1,2] := i2;
                    q1,q2 := min(p1,p2),max(p1,p2);
                    if(r<=q2-q1) then
                        R[N1] := -q1;
                        if(p1<p2) then
                            c1 := 1.0;
                            c2 := 0.0;
                        else
                            c1 := 0.0;
                            c2 := 1.0;
                        end if;
                    else
                        R[N1] := (r*r-2*(p1+p2)*r+(p1-p2)*(p1-p2))/(4*r);
                        c1 := (p2-p1+r)/2/r;
                        c2 := (p1-p2+r)/2/r;
                    end if;
                    for j from 1 to m do
                        W[N1,j] := c1*A[i1,j]+c2*A[i2,j];
                    end do;
                end if;
            end do;
        end do;
    end proc;

    pcech1 := Compiler:-Compile(pcech1);

    pcech := proc(A,pow,a1)
        printf("building the 1-skeleton of the cech complex...\n");
        argl := [A,pow,a1];
        md := module()
        option object;
        export A,pow,a1,E,R,N,m,n1,getadj,getadj1,getnerve,getwits,J;
        local init,N1,W1,J1,J2;
            getadj::static := proc(i,a:=a1)
                i0,n := J1[i],J2[i];
                ans := [];
                for i1 from i0 to i0+n-1 do
                    if(R[i1]<=a) then
                        ans := [op(ans),E[i1,2]];
                    end if;
                end do;
                return ans;
            end proc;
            getadj1::static := proc(i,a,V)
                i0,n := J1[i],J2[i];
                for k from 1 to n do
                    i1 := i0+k-1;
                    if(R[i1]<=a) then
                        V[k] := E[i1,2];
                    end if;
                end do;
                return n;
            end proc;
            getnerve::static := proc()
            option remember;
                X := fplex(N);
                for i from 1 to N do
                    if(pow[i]+a1>=0) then
                        X:-addsimps([i]=-pow[i]);
                    end if;
                end do;
                for i1 from 1 to N1 do
                    sig := [E[i1,1],E[i1,2]];
                    if(sig[1]<sig[2]) then
                        X:-addsimps(sig=R[i1]);
                    end if;
                end do;
                return X;
            end proc;
            getwits::static := proc()
            option remember;
                X := getnerve();
                W := vectab(m);
                il0 := [seq(sig[1],sig=X[0])];
                W:-setelt([seq([i],i=il0)],A[il0,..]);
                il1 := [];
                for i1 from 1 to N1 do
                    if(E[i1,1]<E[i1,2]) then
                        il1 := [op(il1),i1];
                    end if;
                end do;
                W:-setelt([seq([E[i1,1],E[i1,2]],i1=il1)],W1[il1,..]);
                return W;
            end proc;
            init::static := proc()
                A,pow,a1 := args;
                N,m := Dimension(A);
                il := [];
                N1 := 2*pcech0(A,pow,a1,N,m);
                E,J1,J2 := allocla[integer[4]]([N1,2],N,N);
                R,W1 := allocla[float[8]](N1,[N1,m]);
                pcech1(A,pow,a1,E,R,W1,N,m);
                i := 1;
                J1[1] := 1;
                il1 := [];
                for i1 from 1 to N1 do
                    sig := [E[i1,1],E[i1,2]];
                    if(i<>sig[1]) then
                        i := sig[1];
                        J1[i] := i1;
                        n1 := 0;
                    end if;
                    J2[i] := J2[i]+1;
                end do;
                n1 := max(J2);
                J := allocla[integer[4]](n1);
                return;
            end proc;
            init(op(argl));
        end module;
        return md;
    end proc;

    vectab := proc(m,typ)
        if(type(args[1],'Matrix')) then
            A := args[1];
            return vectab([seq(i,i=1..Dimension(A)[1])],A);
        elif(type(args[1],'list')) then
            xl,A := args;
            md := vectab(Dimension(A)[2]);
            md:-setelt(xl,A);
            return md;
        end if;
        argl := [args];
        md := module()
        option object;
        export m,A,N,`?[]`,getelt,getelt1,setelt,addelt,submat,submat1,N1,inds,labs,`whattype`;
        local V1,ModulePrint,ModuleApply,newrow0,setrow0,J;
            ModulePrint::static := proc()
                return nprintf("%d points in R^%d",N,m);
            end proc;
            `whattype`::static := proc()
                return 'VecTab';
            end proc;
            submat::static := proc(xl)
                n := nops(xl);
                if(nargs=1) then
                    ans := allocla([n,m]);
                    submat1(xl,ans);
                    return ans;
                end if;
            end proc;
            submat1::static := proc(xl,A1)
                n := nops(xl);
                for i from 1 to n do
                    J[i] := inds[xl[i]];
                end do;
                copysubmat1(A,J,A1,n,m);
                return;
            end proc;
            ModuleApply::static := proc(x)
                getelt1(x,V1);
                return convert(V1,'list');
            end proc;
            `?[]`::static := proc()
                x := op(args[2]);
                if(nargs=2) then
                    getelt1(x,V);
                    return V;
                elif(nargs=3) then
                    return setelt(x,op(args[3]));
                else
                    error;
                end if;
                return;
            end proc;
            getelt::static := proc(x)
                if(not assigned(inds[x])) then
                    error "not assigned";
                end if;
                i := inds[x];
                V := allocla(m);
                getrow[V](A,i);
                return V;
            end proc;
            getelt1::static := proc(x,V)
                i := inds[x];
                getrow[V](A,i);
                return;
            end proc;
            addelt::static := proc(V)
                newrow0();
                setrow0(N,V);
                return;
            end proc;
            setelt::static := proc(x,V)
                if(type(V,'Matrix')) then
                    return setelt1(args);
                else
                    return setelt0(args);
                end if;
            end proc;
            setelt0::static := proc(x,V)
                if(assigned(inds[x])) then
                    i := inds[x];
                else
                    newrow0();
                    i := N;
                    inds[x] := i;
                    labs := [op(labs),x];
                end if;
                setrow0(i,V);
                return;
            end proc;
            setelt1::static := proc(xl,A1)
                n := nops(xl);
                for i from 1 to n do
                    setelt0(xl[i],getrow(A1,i,V1));
                end do;
                return;
            end proc;
            newrow0::static := proc()
                if(N=N1) then
                    N1 := 2*N1;
                    A1 := allocla[float[8]]([N1,m]);
                    A1[1..N,..] := A[1..N,..];
                    A := A1;
                    J1 := allocla[integer[4]](2*N1);
                    J1[1..N] := J[1..N];
                    J := J1;
                end if;
                N := N+1;
            end proc;
            setrow0::static := proc(i,V)
                if(type(V,'Vector'(datatype=float[8]))) then
                    setrow(A,i,V);
                else
                    for j from 1 to m do
                        A[i,j] := V[j];
                    end do;
                end if;
            end proc;
            m := op(argl);
            labs := [];
            inds := table();
            A,V1 := allocla[float[8]]([8,m],m);
            J := allocla[integer[4]](8);
            N1 := 8;
            N := 0;
        end module;
        return md;
    end proc;

#put the support of a vertex, i.e. the set of edges into the list
#inds1, and the inverse indices into inds2
    pdiag0 := proc(i::integer[4],A::Array(datatype=float[8]),pow::Array(datatype=float[8]),E::Array(datatype=integer[4]),a::float[8],R::Array(datatype=float[8]),inds1::Array(datatype=integer[4]),inds2::Array(datatype=integer[4]),A1::Array(datatype=float[8]),pow1::Array(datatype=float[8]),N::integer[4],m::integer[4],n1::integer[4])
        for k from 1 to N do
            inds2[k] := 0;
        end do;
        l := 0;
        for k from 1 to n1 do
            i1 := E[i,k];
            if(i1=0) then
                break;
            end if;
            if(R[i,k]<=a) then
                l := l+1;
                inds1[l] := i1;
                inds2[i1] := l;
                for j from 1 to m do
                    A1[l,j] := A[i1,j]-A[i,j];
                end do;
                pow1[l] := pow[i1];
            end if;
        end do;
        return l;
    end proc;

    pdiag0 := Compiler:-Compile(pdiag0);

#power diagram with vertices in A, powers in pow, maximum power
#a1. use setcell(i) to set the current cell, then setface(sig) to get the
#weight of [op(sig),i]. only loads equations when the cell is set. n1
#is the maximum number of edges in the star of any vertex, which is
#the maximum number of variables in any inequality.
    pdiag := proc(A,pow,a1)
        argl := [A,pow,a1];
        md := module()
        option object;
        export A,pow,a,a1,N,m,cech,cell,cur,inds1,inds2,wit,setcell,setface,getwit,draw;
        local ModulePrint,init;
            ModulePrint::static := proc()
                s := "power diagram, %d regions in R^%d";
                return nprintf(s,N,m);
            end proc;
            setcell::static := proc(i,b:=a1)
                cur := i;
                n := cech:-getadj1(i,b,inds1);
                invinds1(inds1,inds2,n,N);
                cell:-setcent2(A,pow,i);
                cell:-setsites2(A,pow,inds1,n);
                cell:-loadeqs();
                a := Float(infinity);
                return;
            end proc;
            setface::static := proc(sig,b:=a1)
                sig1 := [seq(inds2[i],i=sig)];
                if(0 in sig1 or not cell:-setface(sig1,b)) then
                    return false;
                end if;
                a := cell:-a;
                return true;
            end proc;
            getwit::static := proc()
                if(a=Float(infinity)) then
                    error "empty face";
                end if;
                return [seq(wit[i],i=1..m)];
            end proc;
            draw::static := proc(a:=a1)
                h := sqrt(max(0.0,a1+max(pow)));
                ans := [drawpd([A,A],pow,a,h)];
                return display(ans);
            end proc;
            init::static := proc()
                A,pow,a1 := args;
                N,m := Dimension(A);
                cech := pcech(A,pow,a1);
                n1 := cech:-n1;
                cell := pcell(n1,m);
                wit := cell:-wit;
                inds1,inds2 := allocla[integer[4]](n1,N);
                cur,n,a := 0,0,Float(infinity);
            end proc;
            init(op(argl));
        end module;
    end proc;

    randpd := proc(N)
        A := Matrix([seq([randf(-5,5),randf(-5,5)],i=1..N)],datatype=float[8]);
        pow := Vector([seq(randf(-1,1),i=1..N)],datatype=float[8]);
        pd := pdiag(A,pow,1.0);
        return pd;
    end proc;

    drawalpha := proc(X,W)
        m := min(W:-m);
        V := allocla[float[8]](m);
        m := min(m,3);
        ans := [];
        for sig in X[1] do
            W:-getelt1([sig[1]],V);
            p1 := [seq(V[i],i=1..m)];
            W:-getelt1([sig[2]],V);
            p2 := [seq(V[i],i=1..m)];
            ans := [op(ans),line(p1,p2,color=black)];
        end do;
        return display(ans);
    end proc;

    alphastep := proc(pd,S)
        N1,l := Dimension(S);
        n,m,a1 := pd:-N,pd:-m,pd:-a1;
        N := 0;
        i := 0;
        T := allocla[integer[4]]([N1,l]);
        aa,W := allocla[float[8]](N1,[N1,m]);
        for i1 from 1 to N1 do
            sig := [seq(S[i1,j],j=1..l)];
            if(sig[1]<>i) then
                i := sig[1];
                pd:-setcell(i);
            end if;
            tprint[10]("  %d/%d vertices, %d simplices...",i,n,N);
            sig1 := sig[2..l];
            if(not pd:-setface(sig1)) then
                next;
            end if;
            wit := pd:-wit;
            a := pd:-a;
            if(a<=a1) then
                N := N+1;
                for j from 1 to l do
                    T[N,j] := sig[j];
                end do;
                aa[N] := a;
                setrow1(W,N,wit,m);
            end if;
        end do;
        return T[1..N],aa[1..N],W[1..N];
    end proc;

#The main loop in the dual alpha complex algorithm. adds the potential
#simplices in S with the desired filtration value to a filtered
#complex X. if specific, adds the representatives to the witness map W
    addalpha := proc(pd,S,X,W)
        N1,l := Dimension(S);
        n := pd:-N;
        a1 := pd:-a1;
        N := 0;
        i := 0;
        for i1 from 1 to N1 do
            sig := [seq(S[i1,j],j=1..l)];
            if(sig[1]<>i) then
                i := sig[1];
                pd:-setcell(i);
            end if;
            tprint[10]("  %d/%d vertices, %d simplices...",i,n,N);
            sig1 := sig[2..l];
            if(not pd:-setface(sig1)) then
                next;
            end if;
            wit := pd:-wit;
            a := pd:-a;
            if(a<=a1) then
                N := N+1;
                j := X:-addfilt(sig,a);
                W:-setelt(j,wit);
            end if;
        end do;
        printf("%d total simplices\n",N);
    end proc;

#computes the alpha complex. may take a power diagram as input, or a
#matrix, power vector, and maximum cutoff. if no power vector is
#specified, sets all to zero and uses norm instead of norm squared to
#be consistent with existing notation
    getalpha := proc(pd,dim)
        if(not type(procname,indexed) or op(procname)=false) then
            return getalpha[true](args)[1];
        end if;
        if(type(args[1],'Matrix')) then
            if(not type(args[2],'Vector')) then
                A,a1 := args[1..2];
                N := Dimension(A)[1];
                return procname(pdiag(A,allocla[float[8]](N),a1^2),args[3]);
            else
                return procname(pdiag(args[1..3]),args[4]);
            end if;
        end if;
        cech := pd:-cech;
        printf("computing the cech graph G...\n");
        X := cech:-getnerve();
        printf("%d vertices, %d edges, max degree=%d\n",X:-numsimps(0),X:-numsimps(1),cech:-n1);
        Y := fplex(pd:-N);
        W := witmap(Y,pd:-m);
        for k from 0 to dim do
            if(k<=1) then
                S := X:-getmat(k);
            else
                printf("computing the %d-dimensional lazy complex...\n",k);
                S := getlazy(Y:-getmat(k-1));
                printf("%d simplices\n",Dimension(S)[1]);
            end if;
            printf("working on the %d-dimensional simplices...\n",k);
            addalpha(pd,S,Y,W);
        end do;
        return Y,W;
    end proc;

#better name
    alphaplex := getalpha;

#computes the witness complex associated to a power diagram and a map
#f on the representative witnesses.
    alphawit := proc(pd,dim,f)
        a1,m := pd:-a1,pd:-m;
        wit := allocla[float[8]](m);
        cech := pd:-cech;
        printf("computing the cech graph G...\n");
        X := cech:-getnerve();
        printf("%d vertices, %d edges, max degree=%d\n",X:-numsimps(0),X:-numsimps(1),cech:-n1);
        Y := fplex(pd:-N);
        W := witmap(Y,pd:-m);
        for k from 0 to dim do
            if(k<=1) then
                S := X:-getmat(k);
            else
                printf("computing the %d-dimensional lazy complex...\n",k);
                S := getlazy(Y:-getmat(k-1));
                printf("%d simplices\n",Dimension(S)[1]);
            end if;
            printf("working on the %d-dimensional simplices...\n",k);
            T,aa,wits := alphastep(pd,S);
            N := Dimension(T)[1];
            for i from 1 to N do
                getrow1(wits,i,wit,m);
                a := f(wit,aa[i]);
                if(a<=a1) then
                    n := Y:-addfilt([seq(T[i,j],j=1..m)],a);
                    W:-setelt(n,wit);
                end if;
            end do;
            Y:-reduce(k);
        end do;
        return Y,W;
    end proc;

#witness map from X to R^m, which sends each simplex to its unique
#minimal representative
    witmap := proc(X,m)
        md := module()
        option object;
        export X,m,A,setelt,getvec,getelt,init;
        local dyn,ModulePrint,ModuleApply,flags;
            ModulePrint::static := proc()
                return nprintf("witness map to R^%d",m);
            end proc;
            ModuleApply::static := proc(sig)
                return getelt(sig);
            end proc;
            getelt::static := proc(sig)
                i := X:-getind(sig);
                if(not flags[i]) then
                    error "witness map not defined";
                end if;
                return [seq(A[i,j],j=1..m)];
            end proc;
            getvec::static := proc(sig)
                i := X:-getind(sig);
                if(not flags[i]) then
                    error "witness map not defined";
                end if;
                return A[X:-getind(sig)];
            end proc;
            setelt::static := proc(i,V)
                if(dyn:-allocif(i)) then
                    A,flags := dyn:-getelts();
                end if;
                flags[i] := true;
                for j from 1 to m do
                    A[i,j] := V[j];
                end do;
                return;
            end proc;
            init::static := proc()
                X,m := args;
                dyn := dynla(float[8](m),boolean);
                A,flags := dyn:-getelts();
                return;
            end proc;
        end module;
        md:-init(X,m);
        return md;
    end proc;

    drawpd := proc(L,pow,r,h)
        L1,L2,G := metpair[op(getsubs(procname))](L,true);
        n,m := Dimension(L1);
        if(nargs=3) then
            h1 := sqrt(max(SingularValues(G,output='S'))*max(pow));
            return drawpd[G]([L1,L2],pow,r,h1);
        end if;
        ans := [];
        if(r<>Float(infinity)) then
            try
                ans := [op(ans),drawpow[G]([L1,L2],pow,r)];
            catch "empty ball diagram":
            end try;
        end if;
        ans := [op(ans),drawsv([L1,L2],pow,[],h,args[4..nargs])];
        return display(ans);
    end proc;

    drawpow := proc(L,pow,r)
        L1,L2,G := metpair[op(getsubs(procname))](L,true);
        n,m := Dimension(L1);
        V := Vector([seq(pow[i]+r,i=1..n)],datatype=float[8]);
        return drawcech[G](L1,V,args[4..nargs]);
    end proc;

#draw the shifted voronoi diagram with vertices given by L, and L2=L1.G the dual
#basis under some inner product given by a symmetric matrix G. taking
#Lt=L gives the usual metric.
    drawsv := proc(L,pow,il,rng)
    uses LinearAlgebra;
    local x,y;
        L1,L2 := metpair[op(getsubs(procname))](L);
        n,m := Dimension(L1);
        if(m<>2) then
            error "wrong dimension for drawing";
        elif(nargs=1) then
            return procname([L1,L2],Vector(n,datatype=float[8]));
        elif(nargs=2) then
            return procname([L1,L2],pow,[]);
        elif(nargs=3) then
            return procname(L,pow,il,0.0);
        elif(type(args[4],'numeric')) then
            return procname(L,pow,il,[drawrange(L1,args[4])]);
        end if;
        rng1,rng2 := op(rng);
        a1,b1 := op(rng1);
        a2,b2 := op(rng2);
        r1 := max(b1-a1,b2-a2);
        A := Matrix(n+3,2,datatype=float[8]);
        P := Matrix(n+3,2,datatype=float[8]);
        V := Vector(n+3,datatype=float[8]);
        ans := [];
        A[n,1],A[n,2],V[n] := -1,0,-a1+.1*r1;
        A[n+1,1],A[n+1,2],V[n+1] := 1,0,b1+.1*r1;
        A[n+2,1],A[n+2,2],V[n+2] := 0,-1,-a2+.1*r1;
        A[n+3,1],A[n+3,2],V[n+3] := 0,1,b2+.1*r1;
        for i from 1 to n do
            N := 0;
            for j from 1 to n do
                if(j=i) then
                    next;
                end if;
                N := N+1;
                u := L2[j,..]-L2[i,..];
                v := (L1[i,..]+L1[j,..])/2;
                b := DotProduct(u,v);
                b := b-(pow[j]-pow[i])/2;
                A[N,1] := u[1];
                A[N,2] := u[2];
                V[N] := b;
            end do;
            pl := li2poly(A,V,n+3,P);
            if(nops(pl)=0) then
                next;
            end if;
            if(i in il) then
                t := .4;
            else
                t := 1.0;
            end if;
            ans := [op(ans),op(linepoly(pl,thickness=3))];
        end do;
        return display(ans,view=[rng1,rng2]);
    end proc;

    linepoly := proc(pl)
        n := nops(pl);
        ans := [];
        for i from 1 to n-1 do
            ans := [op(ans),line(pl[i],pl[i+1],args[2..nargs])];
        end do;
        ans := [op(ans),line(pl[n],pl[1],args[2..nargs])];
        return ans;
    end proc;

    li2poly0 := proc(A::Array(datatype=float[8]),
                     V::Array(datatype=float[8]),
                     N::intger[4],
                     P::Array(datatype=float[8]))
        tol := .000001;
        n := 0;
        for i from 1 to N do
            for j from i+1 to N do
                a11,a12,a21,a22 := A[i,1],A[i,2],A[j,1],A[j,2];
                v1,v2 := V[i],V[j];
                d := a11*a22-a12*a21;
                if(abs(d)<tol) then
                    next;
                end if;
                u1 := (a22*v1-a12*v2)/d;
                u2 := (-a21*v1+a11*v2)/d;
                isint := true;
                for k from 1 to N do
                    if(A[k,1]*u1+A[k,2]*u2>V[k]+tol) then
                        isint := false;
                        break;
                    end if;
                end do;
                if(isint) then
                    n := n+1;
                    P[n,1] := u1;
                    P[n,2] := u2;
                end if;
            end do;
        end do;
        return n;
    end proc;

    li2poly0 := Compiler:-Compile(li2poly0);

    li2poly := proc(A,V,N,P)
        n := li2poly0(A,V,N,P);
        if(n=0) then
            return [];
        end if;
        pl := [seq([P[i,1],P[i,2]],i=1..n)];
        ql := [seq(pl[i]-pl[1],i=2..n)];
        rl := [seq(sqrt(q[1]^2+q[2]^2),q=ql)];
        c0 := 1;
        i0,j0 := 0,0;
        for i from 1 to n-1 do
            for j from i+1 to n-1 do
                c := ql[i][1]*ql[j][1]+ql[i][2]*ql[j][2];
                c := c/rl[i]/rl[j];
                if(c<=c0) then
                    c0 := c;
                    i0,j0 := i,j;
                end if;
            end do;
        end do;
        cl := [seq((ql[i0][1]*ql[i][1]+ql[i0][2]*ql[i][2])/rl[i0]/rl[i],i=1..n-1)];
        sig := sort(cl,`>`,output=permutation);
        return [pl[1],seq(pl[sig[i]+1],i=1..n-1)];
    end proc;

    metpair := proc(A)
        if(type(A,'list')) then
            A1 := A[1];
        else
            A1 := A;
        end if;
        A2 := A1;
        if(type(procname,indexed) and nops(procname)=1) then
            G := op(procname);
            A2 := A2.G;
        end if;
        if(type(A1,'Matrix')) then
            m := Dimension(A1)[2];
        elif(type(A1,'Vector')) then
            m := Dimension(A1);
        end if;
        if(nargs=1 or args[2]=false) then
            return A1,A2;
        end if;
        if(type(procname,indexed) and nops(procname)=1) then
            G := op(procname);
        elif(type(L,'list')) then
            G := (Transpose(L1).L1)^(-1).Transpose(L1).L2;
        else
            G := DiagonalMatrix([seq(1.0,i=1..m)],datatype=float[8],shape=diagonal);
        end if;
        return A1,A2,G;
    end proc;

    drawcech := proc(A,V)
    local x,y;
        if(type(procname,indexed)) then
            G := op(procname);
        else
            G := Matrix(IdentityMatrix(2),datatype=float[8]);
        end if;
        n,m := Dimension(A);
        if(m<>2) then
            error "must be two-dimension to draw a cover";
        end if;
        ans := [];
        for i from 1 to n do
            r := V[i];
            if(r<=0) then
                next;
            end if;
            p := [A[i,1],A[i,2]];
            ans := [op(ans),getellipse(p,G,r,filled=true,linestyle="dash",thickness=0,color=RGB(.7,.7,.7),args[3..nargs])];
        end do;
        for i from 1 to n do
            r := V[i];
            if(r<=0) then
                next;
            end if;
            p := [A[i,1],A[i,2]];
            ans := [getellipse(p,G,r,filled=false,linestyle="dash",thickness=3,color=RGB(0,0,0),args[3..nargs]),op(ans)];
        end do;
        for i from 1 to n do
            if(V[i]<0) then
                next;
            end if;
            u := [A[i,1],A[i,2]];
            ans := [op(ans),textplot([op(u),i],color="blue")];
        end do;
        if(nops(ans)=0) then
            error "empty ball diagram";
        end if;
        return display(ans);
    end proc;

    getellipse := proc(p,G,r)
        U1,S1 := SingularValues(G,output=['U','S']);
        t1 := arctan(U1[2,1]/U1[1,1]);
        p1 := [cos(-t1)*p[1]-sin(-t1)*p[2],sin(-t1)*p[1]+cos(-t1)*p[2]];
        ell1 := ellipse([p1[1],p1[2]],sqrt(r)/sqrt(S1[1]),sqrt(r)/sqrt(S1[2]),args[4..nargs]);
        ell1 := rotate(ell1,t1);
        return ell1;
    end proc;

    powheat0 := proc(A::Array(datatype=float[8]),pow::Array(datatype=float[8]),a1::float[8],x0::float[8],y0::float[8],t::float[8],B::Array(datatype=float[8]),N::integer[4],m::integer[4],n::integer[4])
        for k from 1 to N do
            x := A[k,1];
            y := A[k,2];
            r := sqrt(max(a1+pow[k],0.0));
            i0 := floor(.5+(x-r-x0)/t);
            i1 := ceil(.5+(x+r-x0)/t);
            j0 := floor(.5+(y-r-y0)/t);
            j1 := ceil(.5+(y+r-y0)/t);
            for i from max(i0,1) to min(i1,m) do
                for j from max(j0,1) to min(j1,n) do
                    x1 := x0+(i-.5)*t;
                    y1 := y0+(j-.5)*t;
                    b := (x1-x)*(x1-x)+(y1-y)*(y1-y)-pow[k];
                    if(b<=a1) then
                        B[n-j+1,i] := min(B[n-j+1,i],b);
                    end if;
                end do;
            end do;
        end do;
    end proc;

    powheat0 := Compiler:-Compile(powheat0);

#draw the heat map of a power diagram
    powheat := proc(A,pow,a1,M:=1000)
    local c;
        N := Dimension(A)[1];
        kl := [];
        for k from 1 to N do
            if(-pow[k]<=a1) then
                kl := [op(kl),k];
            end if;
        end do;
        x0 := min(seq(A[k,1]-sqrt(max(a1+pow[k],0.0)),k=kl));
        x1 := max(seq(A[k,1]+sqrt(max(a1+pow[k],0.0)),k=kl));
        y0 := min(seq(A[k,2]-sqrt(max(a1+pow[k],0.0)),k=kl));
        y1 := max(seq(A[k,2]+sqrt(max(a1+pow[k],0.0)),k=kl));
        print(x0,x1,y0,y1);
        t := max((x1-x0)/M,(y1-y0)/M);
        m := ceil((x1-x0)/t);
        n := ceil((y1-y0)/t);
        B := allocla[float[8]]([n,m]);
        ArrayTools:-Fill(Float(infinity),B);
        powheat0(A,pow,a1,x0,y0,t,B,N,m,n);
        c0 := min(B);
        map[inplace](c->(c-c0)/(a1-c0),B);
        return heatmat(B);
    end proc;

end module;
