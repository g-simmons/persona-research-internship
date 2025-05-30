KerDens := module()
option package;
export getkde,drawdens,setdenscol,gaussfit,densmap,densheat,denscut,cdfcut,denscolor;
local ModuleLoad;

    getkde0 := proc(A::Array(datatype=float[8]),h::float[8],aa::Array(datatype=float[8]),V::Array(datatype=float[8]),U::Array(datatype=float[8]),N::integer[4],m::integer[4])
        h2 := 2*h*h;
        for i from 1 to N do
            r := 0.0;
            for j from 1 to m do
                c := (V[j]-A[i,j]);
                r := r+c*c;
            end do;
            U[i] := aa[i]*exp(-r/h2);
        end do;
    end proc;

    getkde0 := Compiler:-Compile(getkde0);

    getkde1 := proc(A::Array(datatype=float[8]),U::Array(datatype=float[8]),mu::Array(datatype=float[8]),N::integer[4],m::integer[4])
        for j from 1 to m do
            c := 0.0;
            for i from 1 to N do
                c := c+U[i]*A[i,j];
            end do;
            mu[j] := c;
        end do;
        c := 0.0;
        for i from 1 to N do
            c := c+U[i];
        end do;
        for j from 1 to m do
            mu[j] := mu[j]/c;
        end do;
    end proc;

    getkde1 := Compiler:-Compile(getkde1);

    getkde2 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),M::integer[4],N::integer[4],m::integer[4])
        for k from 1 to M do
            i := rand() mod N+1;
            for j from 1 to m do
                B[k,j] := B[k,j]+A[i,j];
            end do;
        end do;
    end proc;

    getkde2 := Compiler:-Compile(getkde2);

    getkde3 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),i::integer[4],M::integer[4],N::integer[4],m::integer[4])
        for k from 1 to M do
            for j from 1 to m do
                B[k,j] := B[k,j]+A[i,j];
            end do;
        end do;
    end proc;

    getkde3 := Compiler:-Compile(getkde3);

    getkde := proc(A,h,aa)
        if(nargs=2) then
            return procname(A,h,evalf(1/Dimension(A)[1]));
        elif(type(args[3],'numeric')) then
            eps := args[3];
            N := Dimension(A)[1];
            return procname(A,h,Vector([seq(eps,i=1..N)],datatype=float[8]));
        end if;
        argl := [A,h,aa];
        md := module()
        option object;
        export V,A,h,aa,U,N,m,numpoints,getdim,K,rho,setpoint,getdens,getdens1,maxdens,maxdens1,getmean,getmean1,`whattype`,`numelems`,init,sample;
        local ModuleApply,ModulePrint,ModuleCopy;
            `whattype`::static := proc()
                return 'KDE';
            end proc;
            ModuleCopy::static := proc(f1,f2)
                f1:-init();
                return;
            end proc;
            ModulePrint::static := proc()
            uses StringTools;
                s := cat("KDE in R^%d");
                return nprintf(s,m);
            end proc;
            `numelems`::static := proc()
                return N;
            end proc;
            numpoints::static := proc()
                return N;
            end proc;
            getdim::static := proc()
                return m;
            end proc;
            K::static := proc(r)
                return exp(-r/h^2/2);
            end proc;
            rho::static := proc(x,y)
                return K(add((x[j]-y[j])^2,j=1..m));
            end proc;
            sample::static := proc(B)
            uses Statistics;
                if(type(args[1],'numeric')) then
                    return procname(allocla[float[8]]([args[1],m]));
                end if;
                M := Dimension(B)[1];
                Sample(Normal(0,h),B);
                if(type(procname,indexed)) then
                    i := op(procname);
                    getkde3(A,B,i,M,N,m);
                else
                    getkde2(A,B,M,N,m);
                end if;
                return B;
            end proc;
            setpoint::static := proc(x)
                for j from 1 to m do
                    V[j] := x[j];
                end do;
                getkde0(A,h,aa,V,U,N,m);
                return;
            end proc;
            getdens::static := proc(x)
                setpoint(x);
                return getdens1(U);
            end proc;
            getdens1::static := proc(x)
                return convert(U,`+`);
            end proc;
            maxdens::static := proc(x)
                setpoint(x);
                return maxdens1();
            end proc;
            maxdens1::static := proc(x)
                return max(U);
            end proc;
            getmean::static := proc()
                mu := Vector(m,datatype=float[8]);
                getmean1(mu);
                return mu;
            end proc;
            getmean1::static := proc(mu)
                getkde1(A,U,mu,N,m);
            end proc;
            ModuleApply::static := getdens;
            init::static := proc()
                N,m := Dimension(A);
                U,V := allocla[float[8]](N,m);
                return;
            end proc;
            A,h,aa := op(argl);
            init();
        end module;
        return md;
    end proc;

    densmap := proc(f,d1,rng)
    local a;
        if(type(procname,indexed)) then
            M := op(procname);
        else
            M := 1000;
        end if;
        #h := f:-h;
        #a1 := -2*h^2*log(d1);
        a1 := -2*log(d1);
        x0,x1 := op(rng[1]);
        y0,y1 := op(rng[2]);
        t := min((x1-x0)/M,(y1-y0)/M);
        m := ceil((x1-x0)/t);
        n := ceil((y1-y0)/t);
        B := allocla[float[8]]([n,m]);
        ArrayTools:-Fill(Float(infinity),B);
        for i from 1 to m do
            for j from 1 to n do
                x := x0+(i-.5)*t;
                y := y0+(j-.5)*t;
                d := f([x,y]);
                if(d>d1) then
                    #B[n-j+1,i] := -2*h^2*log(d);
                    B[n-j+1,i] := -2*log(d);
                end if;
            end do;
        end do;
        a0 := min(B);
        #print(a0,a1);
        #print(max(B));
        map[inplace](a->(a-a0)/(a1-a0),B);
        #print(max(B));
        return B;
    end proc;

    denscut := proc(f,s)
        if(not type(procname,indexed)) then
            return denscut['CONFINT'](args);
        end if;
        typ := op(procname);
        A := f:-A;
        N,m := Dimension(A);
        if(eqtype(typ,'CDF')) then
            N1 := 1000;
            A1 := rowsamp(A,N1);
            V := maprows(f,A1);
            sort[inplace](V);
            N2 := ceil(N1*(1-s));
            return V[N2];
        elif(eqtype(typ,'MAXFRAC')) then
            V := maprows(f,f:-A);
            return s*max(V);
        else
            error;
        end if;
    end proc;

    cdfcut := proc(f,s)
        if(type(args[1],'Matrix')) then
            return procname(getkde(args[1..2]),args[3]);
        end if;
        A := f:-A;
        N,m := Dimension(A);
        N1 := 1000;
        A1 := rowsamp(A,N1);
        V := rowmap(f,A1);
        sort[inplace](V);
        N2 := ceil(N1*(1-s));
        return V[N2];
    end proc;

    densheat := proc(A,mindens,cmap:='viridis')
        B := matf(A);
        m1,m2 := Dimension(B);
        for i from 1 to m1 do
            for j from 1 to m2 do
                x := B[i,j];
                if(x<=mindens) then
                    B[i,j] := Float(infinity);
                else
                    B[i,j] := -log(B[i,j]);
                end if;
            end do;
        end do;
        cmap1 := revcolor(colormap(cmap));
        return heatmap(B,cmap1);
    end proc;

    denscolor := proc()
    option remember;
        return 'viridis';
    end proc;

end module;
