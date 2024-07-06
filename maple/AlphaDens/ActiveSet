ActiveSet := module()
option package;
export dualalg,dualalg1,dualalgalloc,kkt,primalqp,primalqp1,primalqp2,dualqp;

    dualalg0 := proc(C::Array(datatype=float[8]),
                     U::Array(datatype=float[8]),
                     N::integer[4],
                     active::Array(datatype=integer[4]),
                     l0::integer[4],
                     l1::integer[4],
                     objmax::float[8],
                     primtol::float[8],
                     singtol::float[8],
                     ll::Array(datatype=float[8]),
                     mm::Array(datatype=float[8]),
                     ll1::Array(datatype=float[8]),
                     C0::Array(datatype=float[8]),
                     R1::Array(datatype=float[8]),
                     U0::Array(datatype=float[8]),
                     D1::Array(datatype=float[8]),
                     p::Array(datatype=float[8]),
                     z::Array(datatype=float[8]),
                     bb::Array(datatype=float[8]))
        #find the length of the active set (probably l0)
        for l from l0+1 to l1 do
            if(active[l]=0) then
                break;
            end if;
        end do;
        l := l-1;
        for a from 1 to l do
            ll[a] := 0.0;
        end do;
        #initialize the variables
        for a from 1 to l do
            ll[a] := 0.0;
        end do;
        #compute the square submatrix of H corresponding to the
        #active set
        for a from 1 to l do
            for b from 1 to l do
                C0[a,b] := C[active[a],active[b]];
            end do;
            U0[a] := U[active[a]];
        end do;
        #compute the cholesky decomposition of C0[1..l0,1..l0]
        for a from 1 to l0 do
            for b from 1 to a-1  do
                R1[a,b] := C0[a,b];
                for k from 1 to b-1 do
                    R1[a,b] := R1[a,b]-R1[a,k]*R1[b,k]*D1[k];
                end do;
                R1[a,b] := 1.0/D1[b]*R1[a,b];
            end do;
            D1[a] := C0[a,a];
            for b from 1 to a-1 do
                D1[a] := D1[a]-R1[a,b]^2*D1[b];
            end do;
            R1[a,a] := 1.0;
        end do;
        #determine if C0 is singular
        for a0 from 1 to l0 do
            if(D1[a0]<singtol) then
                return objmax;
            end if;
        end do;
        #compute the cholesky decomposition of C0[1..l0,1..l0]
        for a from l0+1 to l do
            for b from 1 to a-1  do
                R1[a,b] := C0[a,b];
                for k from 1 to b-1 do
                    R1[a,b] := R1[a,b]-R1[a,k]*R1[b,k]*D1[k];
                end do;
                R1[a,b] := 1.0/D1[b]*R1[a,b];
            end do;
            D1[a] := C0[a,a];
            for b from 1 to a-1 do
                D1[a] := D1[a]-R1[a,b]^2*D1[b];
            end do;
            R1[a,a] := 1.0;
        end do;
        ans := 0.0;
        for a from 1 to l do
            ans := ans+ll[a]*U0[a];
        end do;
        for a from 1 to l do
            for b from 1 to l do
                ans := ans-C0[a,b]*ll[a]*ll[b]/2;
            end do;
        end do;
        #begin main loop
        while(true) do
            #determine if C0 is singular
            for a0 from l0+1 to l do
                if(D1[a0]<singtol) then
                    D1[a0] := 0.0;
                    break;
                end if;
            end do;
            if(a0=l+1) then
                #signal that C0 is nonsingular
                a0 := 0;
            end if;
            if(a0=0) then
                #C0 is nonsingular. solve C0.ll1=U0
                for a from 1 to l do
                    ll1[a] := U0[a];
                end do;
                for a from 1 to l do
                    for b from 1 to a-1 do
                        ll1[a] := ll1[a]-R1[a,b]*ll1[b];
                    end do;
                end do;
                for a from 1 to l do
                    ll1[a] := ll1[a]/D1[a];
                end do;
                for a from l to 1 by -1 do
                    for b from a+1 to l do
                        ll1[a] := ll1[a]-R1[b,a]*ll1[b];
                    end do;
                end do;
                #find the smallest index and minimum value pair of
                #the elements in ll1, i.e. the most violated
                #constraint
                a0,c0 := 0,0.0;
                for a from l0+1 to l do
                    if(ll1[a]<c0) then
                        a0 := a;
                        c0 := ll1[a];
                    end if;
                end do;
                if(a0=0) then
                    #we are in the feasible region. update ll,ans1
                    for a from 1 to l do
                        ll[a] := ll1[a];
                    end do;
                    ans1 := 0.0;
                    for a from 1 to l do
                        ans1 := ans1+ll[a]*U0[a];
                    end do;
                    for a from 1 to l do
                        for b from 1 to l do
                        ans1 := ans1-C0[a,b]*ll[a]*ll[b]/2;
                        end do;
                end do;
                else
                    for a from 1 to l do
                    p[a] := ll1[a]-ll[a];
                    end do;
                end if;
            else
                #solve Transpose(R1[1..a0-1,1..a0-1]).p[1..a0-1]=-R1[a0,1..a0-1]
                for a from 1 to a0-1 do
                    p[a] := -R1[a0,a];
                end do;
                for a from a0-1 to 1 by -1 do
                    for b from a+1 to a0-1 do
                        p[a] := p[a]-R1[b,a]*p[b];
                    end do;
                end do;
                p[a0] := 1.0;
                for a from a0+1 to l do
                    p[a] := 0.0;
                end do;
                c2 := 0.0;
                for a from 1 to l do
                    c2 := c2+p[a]*U0[a];
                end do;
                if(c2<0.0) then
                    error "1";
                end if;
            end if;
            #if a0>0 then we need to move in the direction of p until a
            #wall is hit
            if(a0>0) then
            #find the coordinate which becomes negative first
                #along the ray ll+t*p
                bounded := false;
                for a from l0+1 to l do
                    if(p[a]>=0.0) then
                        next;
                    end if;
                    t1 := -ll[a]/p[a];
                    if(not bounded) then
                        t := t1;
                        a0 := a;
                        bounded := true;
                    elif(t1<t) then
                        t := t1;
                        a0 := a;
                    end if;
                end do;
                if(t<0) then
                    error "2";
                end if;
                if(not bounded) then
                    return objmax;
                end if;
                for a from 1 to l do
                    if(a=a0) then
                        ll[a] := 0.0;
                    else
                        ll[a] := ll[a]+t*p[a];
                    end if;
                end do;
                #update C0,U0,ans1;
                aa := D1[a0];
                for b from a0+1 to l do
                    z[b] := R1[b,a0];
                end do;
                for b from a0+1 to l do
                    D1[b-1] := D1[b]+aa*z[b]^2;
                    bb[b] := z[b]*aa/D1[b-1];
                    aa := D1[b]*aa/D1[b-1];
                    for r from b+1 to l do
                        z[r] := z[r]-z[b]*R1[r,b];
                        R1[r-1,b-1] := R1[r,b]+bb[b]*z[r];
                    end do;
                    for r from 1 to a0-1 do
                        R1[b-1,r] := R1[b,r];
                    end do;
                end do;
                l := l-1;
                for a from a0 to l do
                    active[a] := active[a+1];
                    ll[a] := ll[a+1];
                end do;
                active[l+1] := 0;
                ll[l+1] := 0.0;
                for a from a0 to l do
                    for b from 1 to a0-1 do
                        C0[a,b] := C0[a+1,b];
                        C0[b,a] := C0[a,b];
                    end do;
                    for b from a0 to l do
                        C0[a,b] := C0[a+1,b+1];
                    end do;
                end do;
                for a from a0 to l do
                    U0[a] := U0[a+1];
                end do;
                #update C0,U0,ans
                for a from 1 to l do
                    for b from 1 to l do
                        C0[a,b] := C[active[a],active[b]];
                    end do;
                    U0[a] := U[active[a]];
                end do;
                #compute the value of the objective function
                ans1 := 0.0;
                for a from 1 to l do
                    ans1 := ans1+ll[a]*U0[a];
                end do;
                for a from 1 to l do
                    for b from 1 to l do
                        ans1 := ans1-C0[a,b]*ll[a]*ll[b]/2;
                    end do;
                end do;
            else
                #test to see if we have an optimal solution
                #using the lagrange multipliers mm
                k,c1 := 0,0.0;
                for i from 1 to N do
                    mm[i] := -U[i];
                    for b from 1 to l do
                        mm[i] := mm[i]+C[i,active[b]]*ll[b];
                    end do;
                    if(mm[i]<c1) then
                        c1 := mm[i];
                        k := i;
                    end if;
                end do;
                if(c1>=-primtol or l=l1) then
                    #we have found a global optimum or are about to exceed
                    #l1
                    return ans1;
                else
                    l := l+1;
                    active[l] := k;
                    ll[l] := 0.0;
                    #update C0,U0
                    for a from 1 to l do
                        C0[a,l] := C[active[l],active[a]];
                        C0[l,a] := C0[a,l];
                    end do;
                    U0[l] := U[active[l]];
                    for a from 1 to l-1 do
                        z[a] := C0[a,l];
                    end do;
                    for a from 1 to l-1 do
                        for b from 1 to a-1 do
                            z[a] := z[a]-R1[a,b]*z[b];
                        end do;
                    end do;
                    for a from 1 to l-1 do
                        z[a] := z[a]/D1[a];
                    end do;
                    for b from 1 to l-1 do
                        R1[l,b] := z[b];
                    end do;
                    R1[l,l] := 1.0;
                    D1[l] := C0[l,l];
                    for a from 1 to l-1 do
                        D1[l] := D1[l]-z[a]^2*D1[a];
                    end do;
                end if;
            end if;
            if(ans1>=objmax) then
                return ans1;
            elif(ans1<ans) then
                #really should return the answer here
                printf("%f,%f,%f\n",ans1,ans,objmax);
                error "3";
            end if;
            ans := ans1;
        end do;
    end proc;

    dualalg1 := proc(C,U,N,active,l0,l1,objmax,primtol,singtol,ll,mm,ll1,C0,R1,U0,D1,p,z,bb)
        print("dual active set algorithm verbose version");
        #initialize the variables
        #find the length of the starting active set (probably l0)
        for l from l0+1 to l1 do
            if(active[l]=0) then
                break;
            end if;
        end do;
        l := l-1;
        for a from 1 to l do
            ll[a] := 0.0;
        end do;
        #compute the square submatrix of H corresponding to the
        #active set
        for a from 1 to l do
            for b from 1 to l do
                C0[a,b] := C[active[a],active[b]];
            end do;
            U0[a] := U[active[a]];
        end do;
        #compute the cholesky decomposition of C0
        for a from 1 to l0 do
            for b from 1 to a-1  do
                R1[a,b] := C0[a,b];
                for k from 1 to b-1 do
                    R1[a,b] := R1[a,b]-R1[a,k]*R1[b,k]*D1[k];
                end do;
                R1[a,b] := 1.0/D1[b]*R1[a,b];
            end do;
            D1[a] := C0[a,a];
            for b from 1 to a-1 do
                D1[a] := D1[a]-R1[a,b]^2*D1[b];
            end do;
            R1[a,a] := 1.0;
        end do;
        #determine if C0 is singular
        for a0 from 1 to l0 do
            if(D1[a0]<singtol) then
                return objmax;
            end if;
        end do;
        #compute the square submatrix of H corresponding to the
        #active set
        for a from 1 to l do
            for b from 1 to l do
                C0[a,b] := C[active[a],active[b]];
            end do;
            U0[a] := U[active[a]];
        end do;
        #compute the cholesky decomposition of C0 for the remaining values
        for a from l0+1 to l do
            for b from 1 to a-1  do
                R1[a,b] := C0[a,b];
                for k from 1 to b-1 do
                    R1[a,b] := R1[a,b]-R1[a,k]*R1[b,k]*D1[k];
                end do;
                R1[a,b] := 1.0/D1[b]*R1[a,b];
            end do;
            D1[a] := C0[a,a];
            for b from 1 to a-1 do
                D1[a] := D1[a]-R1[a,b]^2*D1[b];
            end do;
            R1[a,a] := 1.0;
        end do;
        ans := 0.0;
        for a from 1 to l do
            ans := ans+ll[a]*U0[a];
        end do;
        for a from 1 to l do
            for b from 1 to l do
                ans := ans-C0[a,b]*ll[a]*ll[b]/2;
            end do;
        end do;
        #begin main loop
        while(true) do
            #determine if C0 is singular
            print("starting main loop",active[1..l]);
            for a0 from l0+1 to l do
                if(D1[a0]<singtol) then
                    D1[a0] := 0.0;
                    break;
                end if;
            end do;
            if(a0=l+1) then
                #signal that C0 is nonsingular
                a0 := 0;
            end if;
            if(a0=0) then
                #C0 is nonsingular. solve C0.ll1=U0
                print("nonsingular",active[1..l],D1[1..l]);
                for a from 1 to l do
                    ll1[a] := U0[a];
                end do;
                for a from 1 to l do
                    for b from 1 to a-1 do
                        ll1[a] := ll1[a]-R1[a,b]*ll1[b];
                    end do;
                end do;
                for a from 1 to l do
                    ll1[a] := ll1[a]/D1[a];
                end do;
                for a from l to 1 by -1 do
                    for b from a+1 to l do
                        ll1[a] := ll1[a]-R1[b,a]*ll1[b];
                    end do;
                end do;
                #find the smallest index and minimum value pair of
                #the elements in ll1, i.e. the most violated
                #constraint
                a0,c0 := 0,0.0;
                for a from l0+1 to l do
                    if(ll1[a]<c0) then
                        a0 := a;
                        c0 := ll1[a];
                    end if;
                end do;
                if(a0=0) then
                    print("feasible",active[1..l]);
                    #we are in the feasible region. update ll,ans1
                    for a from 1 to l do
                        ll[a] := ll1[a];
                    end do;
                    ans1 := 0.0;
                    for a from 1 to l do
                        ans1 := ans1+ll[a]*U0[a];
                    end do;
                    for a from 1 to l do
                        for b from 1 to l do
                        ans1 := ans1-C0[a,b]*ll[a]*ll[b]/2;
                        end do;
                    end do;
                    print("objective value",active[1..l],ans1);
                else
                    print("infeasible",active[1..l],active[a0]);
                    for a from 1 to l do
                        p[a] := ll1[a]-ll[a];
                    end do;
                end if;
            else
                #solve
                #Transpose(R1[1..a0-1,1..a0-1]).p[1..a0-1]=-R1[a0,1..a0-1]
                print("singular",active[1..l],active[a0]);
                for a from 1 to a0-1 do
                    p[a] := -R1[a0,a];
                end do;
                for a from a0-1 to 1 by -1 do
                    for b from a+1 to a0-1 do
                        p[a] := p[a]-R1[b,a]*p[b];
                    end do;
                end do;
                p[a0] := 1.0;
                for a from a0+1 to l do
                    p[a] := 0.0;
                end do;
                c2 := 0.0;
                for a from 1 to l do
                    c2 := c2+p[a]*U0[a];
                end do;
                if(c2<0.0) then
                    error;
                end if;
            end if;
            #if a0>0 then we need to move in the direction of p until a
            #wall is hit
            if(a0>0) then
                print("correcting",active[1..l],p[1..l]);
                #find the coordinate which becomes negative first
                #along the ray ll+t*p
                t := 0.0;
                bounded := false;
                for a from l0+1 to l do
                    if(p[a]>=0.0) then
                        next;
                    end if;
                    t1 := -ll[a]/p[a];
                    if(not bounded) then
                        t := t1;
                        a0 := a;
                        bounded := true;
                    elif(t1<t) then
                        t := t1;
                        a0 := a;
                    end if;
                end do;
                if(t<0) then
                    error;
                end if;
                if(not bounded) then
                    print("unbounded, returning max value");
                    return objmax;
                end if;
                for a from 1 to l do
                    if(a=a0) then
                        ll[a] := 0.0;
                    else
                        ll[a] := ll[a]+t*p[a];
                    end if;
                end do;
                #update C0,U0,ans1;
                print("fixing coordinate",active[1..l]);
                aa := D1[a0];
                for b from a0+1 to l do
                    z[b] := R1[b,a0];
                end do;
                for b from a0+1 to l do
                    D1[b-1] := D1[b]+aa*z[b]^2;
                    bb[b] := z[b]*aa/D1[b-1];
                    aa := D1[b]*aa/D1[b-1];
                    for r from b+1 to l do
                        z[r] := z[r]-z[b]*R1[r,b];
                        R1[r-1,b-1] := R1[r,b]+bb[b]*z[r];
                    end do;
                    for r from 1 to a0-1 do
                        R1[b-1,r] := R1[b,r];
                    end do;
                end do;
                l := l-1;
                for a from a0 to l do
                    active[a] := active[a+1];
                    ll[a] := ll[a+1];
                end do;
                active[l+1] := 0;
                ll[l+1] := 0.0;
                for a from a0 to l do
                    for b from 1 to a0-1 do
                        C0[a,b] := C0[a+1,b];
                        C0[b,a] := C0[a,b];
                    end do;
                    for b from a0 to l do
                        C0[a,b] := C0[a+1,b+1];
                    end do;
                end do;
                for a from a0 to l do
                    U0[a] := U0[a+1];
                end do;
                #update C0,U0,ans
                for a from 1 to l do
                    for b from 1 to l do
                        C0[a,b] := C[active[a],active[b]];
                    end do;
                    U0[a] := U[active[a]];
                end do;
                #compute the value of the objective function
                ans1 := 0.0;
                for a from 1 to l do
                    ans1 := ans1+ll[a]*U0[a];
                end do;
                for a from 1 to l do
                    for b from 1 to l do
                        ans1 := ans1-C0[a,b]*ll[a]*ll[b]/2;
                    end do;
                end do;
                print("objective value,%d,%d,%d,%f\n",active[1],active[2],active[3],ans1);
            else
                #test to see if we have an optimal solution
                #using the lagrange multipliers mm
                print("was feasible, testing optimality",active[1..l]);
                k,c1 := 0,0.0;
                for i from 1 to N do
                    mm[i] := -U[i];
                    for b from 1 to l do
                        mm[i] := mm[i]+C[i,active[b]]*ll[b];
                    end do;
                    if(mm[i]<c1) then
                        c1 := mm[i];
                        k := i;
                    end if;
                end do;
                if(c1>=-primtol or l=l1) then
                    print("optimum found",active[1..l]);
                    #we have found a global optimum or are about to exceed
                    #l1
                    #print(ll,mm);
                    return ans1;
                else
                    print("negative lagrange multiplier",active[1..l],k,c1);
                    l := l+1;
                    active[l] := k;
                    ll[l] := 0.0;
                    #update C0,U0
                    for a from 1 to l do
                        C0[a,l] := C[active[l],active[a]];
                        C0[l,a] := C0[a,l];
                    end do;
                    U0[l] := U[active[l]];
                    for a from 1 to l-1 do
                        z[a] := C0[a,l];
                    end do;
                    for a from 1 to l-1 do
                        for b from 1 to a-1 do
                            z[a] := z[a]-R1[a,b]*z[b];
                        end do;
                    end do;
                    for a from 1 to l-1 do
                        z[a] := z[a]/D1[a];
                    end do;
                    for b from 1 to l-1 do
                        R1[l,b] := z[b];
                    end do;
                    R1[l,l] := 1.0;
                    D1[l] := C0[l,l];
                    for a from 1 to l-1 do
                        D1[l] := D1[l]-z[a]^2*D1[a];
                    end do;
                    print("added to active set",active[1..l]);
                end if;
            end if;
            if(ans1>=objmax) then
                return ans1;
            elif(ans1<ans) then
                #really should return the answer here
                print("answer got smaller",ans1,ans);
                error;
            end if;
            ans := ans1;
        end do;
    end proc;

    dualalg := Compiler:-Compile(dualalg0);
    #for debugging, comment out the previous line and uncomment this:
    #dualalg := dualalg1;

    kkt0 := proc(X0::Array(datatype=float[8]),
                 m::integer[4],
                 A::Array(datatype=float[8]),
                 active::Array(datatype=integer[4]),
                 lmax::integer[4],
                 ll::Array(datatype=float[8]),
                 X::Array(datatype=float[8]))
        for i from 1 to m do
            for k from 1 to lmax do
                if(active[k]=0) then
                    break;
                end if;
            end do;
            k := k-1;
            c := X0[i];
            for j from 1 to k do
                c := c-A[active[j],i]*ll[j];
            end do;
            X[i] := c;
        end do;
        return k;
    end proc;

    kkt := Compiler:-Compile(kkt0);

#allocate the variables
    dualalgalloc := proc(N,l)
        ll := Vector(l,datatype=float);
        mm := Vector(N,datatype=float);
        ll1 := Vector(l,datatype=float);
        C0 := Matrix(l,l,datatype=float);
        R1 := Matrix(l,l,datatype=float);
        U0 := Vector(l,datatype=float);
        D1 := Vector(l,datatype=float);
        p := Vector(l,datatype=float);
        z := Vector(l,datatype=float);
        bb := Vector(l,datatype=float);
        return [ll,mm,ll1,C0,R1,U0,D1,p,z,bb];
    end proc;

#solution to the quadratic program
#minimize:
#1/2*Transpose(X-X0)H(X-X0)
#subject to:
#A[i,..]HX<=V[i] for not i in il, and
#A[i,..]HX=V[i] for i in il
    primalqp := proc(H,X0,A,V,il)
    uses Optimization;
        N,m := Dimension(A);
        jl := [op({seq(i,i=1..N)} minus {op(il)})];
        A1 := A[jl,..];
        A2 := A[il,..];
        V1 := V[jl];
        V2 := V[il];
        r,X := op(QPSolve([-H.X0,H],[A1.H,V1,A2.H,V2]));
        r := r+Transpose(X0).H.X0/2;
        return X,r;
    end proc;

    #same result as the above, using the dual active set
#method of arnstrom, bemporad and axehill. if objmax is a number that
#is not so big, the function will terminate early. if l1 is not
#equal to the total number of constraints, the algorithm will break
#when the size of the active set exceeds that number.
    primalqp1 := proc(H,X0,A,V,il)
        N,m := Dimension(A);
        objmax := 1000000.0;
        l1 := N;
        l := l1;
        ll,mm,ll1,C0,R1,U0,D1,p,z,bb := op(dualalgalloc(N,l));
        C := Matrix(A.H.Transpose(A),datatype=float[8]);
        U := Vector(A.H.X0-V,datatype=float[8]);
        active := Vector(l,datatype=integer[4]);
        l0 := nops(il);
        for i from 1 to l0 do
            active[i] := il[i];
        end do;
        primtol := .000001;
        singtol := .00000001;
        r := dualalg(C,U,N,active,l0,l1,objmax,primtol,singtol,ll,mm,ll1,C0,R1,U0,D1,p,z,bb);
        X := Vector(m,datatype=float[8]);
        k := kkt(X0,m,A,active,l1,ll,X);
        print(nprintf(cat("active set: ",convert(convert(active[1..k],'list'),'string'))));
        return X,r;
    end proc;

    #method of arnstrom, bemporad and axehill. if objmax is a number that
#is not so big, the function will terminate early. if l1 is not
#equal to the total number of constraints, the algorithm will break
#when the size of the active set exceeds that number.
    primalqp2 := proc(H,X0,A,V,il)
        N,m := Dimension(A);
        objmax := 1000000.0;
        l0 := nops(il);
        active := Vector([op(il),seq(0,k=l0+1..N)],datatype=integer[4]);
        C := Matrix(A.H.Transpose(A),datatype=float[8]);
        U := Vector(A.H.X0-V,datatype=float[8]);
        X := Vector(m,datatype=float[8]);
        r := dualqp(C,U,active,objmax,A,X0,X);
        #print(nprintf(cat("active set: ",convert(convert(active[1..k],'list'),'string'))));
        return X,r;
    end proc;

    dualqp := proc(C,U,active,objmax,A,X0,X)
        N := Dimension(C)[1];
        l1 := Dimension(active);
        if(type(procname,indexed)) then
            ll,mm,ll1,C0,R1,U0,D1,p,z,bb := op(procname);
        else
            ll,mm,ll1,C0,R1,U0,D1,p,z,bb := op(dualalgalloc(N,l1));
        end if;
        for l from 1 to l1 do
            if(active[l]=0) then
                break;
            end if;
        end do;
        l0 := l-1;
        primtol := .000001;
        singtol := .00000001;
        r := dualalg(C,U,N,active,l0,l1,objmax,primtol,singtol,ll,mm,ll1,C0,R1,U0,D1,p,z,bb);
        if(nargs=7) then
            m := Dimension(X0);
            k := kkt(X0,m,A,active,l1,ll,X);
        end if;
        return r;
    end proc;

    dualqp := proc(C,U,active,objmax,A,X0,X)
        if(type(C,'list')) then
            C1,N := op(C);
        else
            C1 := C;
            N := Dimension(C)[1];
        end if;
        l1 := Dimension(active);
        if(type(procname,indexed)) then
            ll,mm,ll1,C0,R1,U0,D1,p,z,bb := op(procname);
        else
            ll,mm,ll1,C0,R1,U0,D1,p,z,bb := op(dualalgalloc(N,l1));
        end if;
        for l from 1 to l1 do
            if(active[l]=0) then
                break;
            end if;
        end do;
        l0 := l-1;
        primtol := .000001;
        singtol := .00000001;
        r := dualalg(C1,U,N,active,l0,l1,objmax,primtol,singtol,ll,mm,ll1,C0,R1,U0,D1,p,z,bb);
        if(nargs=7) then
            m := Dimension(X0);
            k := kkt(X0,m,A,active,l1,ll,X);
        end if;
        return r;
    end proc;

end module;

