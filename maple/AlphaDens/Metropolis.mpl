Metropolis := module()
option package;
export plotpath,bayesmodel,bayesmet,gengraph,drawising,sampising,fishermap,metmodel,gaussdist;
local getvars,metlossc,metwalkc,bayeswalkc;

#convert a symbolic expression in a collection of model parameters and
#variables into a compiled function
    metlossc := proc(model,vars,params)
    local xx,aa;
        l := nops(params);
        m := nops(vars);
        parvals := allocla[float[8]](l);
        params1 := params;
        for i from 1 to l do
            param := params[i];
            if(type(param,`=`)) then
                params1[i] := op(1,param);
                parvals[i] := op(2,param);
            end if;
        end do;
        F := model;
        F := eval(F,[seq(params1[i]=aa[i],i=1..l)]);
        F := eval(F,[seq(vars[i]=xx[i],i=1..m)]);
        s := convert(F,'string');
        s := cat("proc(xx::Array(datatype=float[8]),aa::Array(datatype=float[8])) return ",s,"; end proc;\n");
        f := parse(s);
        return Compiler:-Compile(f),params1,parvals;
    end proc;

    #object for sampling from a gaussian proposal distribution
    gaussdist := proc(m)
        md := module()
        option object;
        export m,sample,init,V,L,plotsamp,setmean,setcov,setspread;
        local ModulePrint;
            ModulePrint::static := proc()
                return nprintf("proposal distribution sampler in R^%d",m);
            end proc;
            sample::static := proc(B,h:=1.0)
                if(type(B,'numeric')) then
                    return sample(matf(args[1],m),h);
                end if;
                Sample(Normal(0,h),B);
                MatrixMatrixMultiply(B,L,inplace);
                normdata[true](B,-V);
                return B;
            end proc;
            setmean::static := proc(V1)
                setvec(V,V1);
            end proc;
            setcov::static := proc(C)
                if(type(C,'numeric')) then
                    r := C;
                    return r*IdentityMatrix(m,datatype=float[8]);
                end if;
                L[..] := Transpose(LUDecomposition(C,method='Cholesky'));
                return;
            end proc;
            setspread::static := proc(A)
                setmean(meanstd(A)[1]);
                setcov(covmat(A));
            end proc;
            init::static := proc()
                m := args;
                L := matf(IdentityMatrix(m));
                V := vecf(m);
                N := 0;
            end proc;
        end module;
        md:-init(m);
        return md;
    end proc;

    metwalkc := proc(loss,vars,params)
    local xx,xx1,aa;
        l := nops(params);
        m := nops(vars);
        loss1 := eval(loss,[seq(params[i]=aa[i],i=1..l)]);
        L := eval(loss1,[seq(vars[i]=xx[i],i=1..m)]);
        L1 := eval(loss1,[seq(vars[i]=xx1[i],i=1..m)]);
        code := "proc(xx::Array(datatype=float[8]),aa::Array(datatype=float[8]),B::Array(datatype=float[8]),beta::float[8],J::Array(datatype=integer[4]),xx1::Array(datatype=float[8]),N::integer[4],m::integer[4])\n";
        code := cat(code,"L:= ",convert(L,'string'),";\n",
                    "n := 0;\n",
                    "for t from 1 to N do\n",
                    "for j from 1 to m do\n",
                    "xx1[j] := xx[j]+B[t,j];\n",
                    "B[t,j] := xx1[j];\n",
                    "end do;\n");
        code := cat(code,"L1 := ",convert(L1,'string'),";\n");
        code := cat(code,"r := (rand() mod 10000000)/evalf(10000000);\n",
                    "if(r<=exp(-beta*(L1-L))) then\n L := L1;\n",
                    "for j from 1 to m do\n xx[j] := xx1[j];\n",
                    "end do;\n n := n+1;\n J[n] := t;\n end if;\n",
                    "end do;\n",
                    "return n;\n",
                    "end proc;\n");
        return Compiler:-Compile(parse(code));
    end proc;

    #uses the metropolis-hastings algorithm to generate randoms walks from
#a given starting point
    metmodel := proc(lm,beta:=1.0);
        if(nops(lm)=2) then
            return metmodel([op(lm),[]],beta);
        end if;
        md := module()
        option object;
        export B,m,l,loss,getloss,getprob,init,vars,params,setparams,aa,beta,getpath,walk,getwalk,sample,g0,g,J,getcode;
        local dyn,lossc,walkc,ModulePrint,ModuleApply,xx0,xx1;;
            ModulePrint::static := proc()
                return nprintf("metropolis-hastings walk in R^%d, beta=%f",m,beta,h);
            end proc;
            getloss::static := proc(xx)
                setvec(xx0,xx);
                return lossc(xx0,aa);
            end proc;
            getprob::static := proc(xx)
                return exp(-beta*getloss(xx));
            end proc;
            walk::static := proc(xx,h,nsteps)
                allocif(nsteps);
                B := g:-sample(nsteps,h);
                return walkc(xx,aa,B,beta,J,xx1,nsteps,m);
            end proc;
            getwalk::static := proc(xx,h,nsteps)
                setvec(xx0,xx);
                walk(xx0,args[2..nargs]);
                return xx0;
            end proc;
            setparams::static := proc(bb)
                setvec(aa,bb);
                return;
            end proc;
            ModuleApply::static := getwalk;
            sample::static := proc(N,h,nsteps)
            local xx;
                B1 := g0:-sample(N);
                return rowmap(xx->getwalk(xx,h,nsteps),B1);
            end proc;
            getpath::static := proc(xx,h,nsteps)
                setvec(xx0,xx);
                n := walk(xx0,h,nsteps);
                return B[[seq(J[i],i=1..n)]];
            end proc;
            setparams := proc(aa1)
                setvec(aa,aa1);
            end proc;
            getparams::static := proc()
                return aa;
            end proc;
            allocif::static := proc(nsteps)
                if(dyn:-allocif(nsteps)) then
                    B,J := dyn:-getelts();
                end if;
            end proc;
            init::static := proc()
                loss,vars,params,beta := args;
                m := nops(vars);
                lossc,params,aa := metlossc(loss,vars,params);
                l := nops(params);
                xx0, xx1 := allocla[float[8]](m,m);
                walkc := metwalkc(loss,vars,params);
                dyn := dynla(float[8](m),integer[4]);
                g0 := gaussdist(m);
                g := gaussdist(m);
                B,J := dyn:-getelts();
            end proc;
        end module;
        md:-init(op(lm),beta);
        return md;
    end proc;

    plotpath := proc(A,dim,col)
        if(type(procname,indexed)) then
            el := [op(procname)];
        else
            el := [];
        end if;
        N,m := Dimension(A);
        if(nargs=1) then
            return procname(A,m);
        elif(nargs=2) then
            return procname(A,dim,'viridis');
        end if;
        cmap := colormap(col);
        return display([seq(point([seq(A[i,j],j=1..dim)],color=cmap(evalf((i-.5)/N)),symbol=solidcircle,symbolsize=4,op(el)),i=1..N)]);
    end proc;

    getvars := proc(f)
        varl := indets(f);
        ans := {};
        for var in varl do
            if(type(var,'symbol')) then
                ans := {op(ans),var};
            end if;
        end do;
        return ans;
    end proc;

#maps aa to it's vector of values of the loss function. has the
#property that the pullback of the standard metric is the kl-distance
#form, whose local value is the fisher information matrix
    fishermap := proc(ba,aa)
    local aa1;
        N := ba:-N;
        U := vecf(N);
        if(type(aa,'Vector')) then
            return fishermap0(ba,aa,U);
        elif(type(aa,'Matrix')) then
            B := aa;
            ans := rowmap(aa1->fishermap0(ba,aa1,U),B);
            if(nargs=3) then
                M := args[3];
                R := PCA(ans);
                ans := R:-principalcomponents[..,1..M];
            end if;
            return ans;
        end if;
    end proc;

    fishermap0 := proc(ba,aa,U)
        N := ba:-N;
        for i from 1 to N do
            U[i] := ba:-getloss(aa,i);
        end do;
        return U;
    end proc;

    bayeswalkc := proc(loss,lvars,lparams)
    local xx,aa,aa1;
        lm0 := eval(loss,[seq(lvars[i]=xx[i],i=1..nops(lvars))]);
        lm := eval(lm0,[seq(lparams[i]=aa[i],i=1..nops(lparams))]);
        lm1 := eval(lm0,[seq(lparams[i]=aa1[i],i=1..nops(lparams))]);
        code := "proc(A::Array(datatype=float[8]),aa::Array(datatype=float[8]),beta::float[8],B::Array(datatype=float[8]),J::Array(datatype=integer[4]),xx::Array(datatype=float[8]),aa0::Array(datatype=float[8]),aa1::Array(datatype=float[8]),N::integer[4],m::integer[4],l::integer[4],nsteps::integer[4])\n";
        code := cat(code,"H := 0.0;\n",
                    "for k from 1 to N do\n",
                    "for j from 1 to m do\n",
                    "xx[j] := A[k,j];\n",
                    "end do;\n",
                    "H := H+(",convert(lm,'string'),");\n",
                    "end do;\n",
                    "n := 0;\n",
                    "for t from 1 to nsteps do\n",
                    "for j from 1 to l do\n",
                    "aa1[j] := B[t,j]+aa[j];\n",
                    "B[t,j] := aa1[j];\n",
                    "end do;\n",
                    "H1 := 0.0;\n",
                    "for k from 1 to N do\n",
                    "for j from 1 to m do\n",
                    "xx[j] := A[k,j];\n",
                    "end do;\n",
                    "H1 := H1+(",convert(lm1,'string'),");\n",
                    "end do;\n",
                    "r := (rand() mod 10000000)/evalf(10000000);\n",
                    "if(r<=exp(-beta*(H1-H))) then \n",
                    "for j from 1 to l do \n",
                    "aa[j] := aa1[j];\n",
                    "end do;\n",
                    "H := H1;\n",
                    "n := n+1;\n",
                    "J[n] := t;\n",
                    "end if;\n",
                    "end do;\n",
                    "return n;\n",
                    "end proc;\n");
        return Compiler:-Compile(parse(code));
    end proc;

#generate walks using bayesian inference to sample models
    bayesmet := proc(lm,A,beta:=1.0)
        md := module()
        option object;
        export lm,A,N,d,m,l,beta,walk,getloss,getprob,getpath,getwalk,getmodel,sample,init,B,J,g,g0;
        local ModulePrint,lossc,xx0,xx1,aa0,aa1,cc,dyn,walkc,allocif;
            ModulePrint::static := proc()
                return nprintf("bayesian learning machine walk, %d parameters, "
                               "%d points in R^%d",l,N,m);
            end proc;
            getloss::static := proc(aa,k)
                if(nargs=2 and type(args[2],'numeric')) then
                    setvec(aa0,aa);
                    getrow1(A,k,xx0,m);
                    return lossc(xx0,aa0);
                elif(nargs=1) then
                    return add(getloss(aa,i),i=1..N);
                end if;
            end proc;
            getmodel::static := proc(aa)
                met := metmodel(lm,beta);
                if(nargs=1) then
                    met:-setparams(aa);
                end if;
                return met;
            end proc;
            getprob::static := proc(aa)
                return exp(-beta*getloss(aa));
            end proc;
            walk::static := proc(aa,h,nsteps)
                allocif(nsteps);
                g:-sample(B,h);
                return walkc(A,aa,beta,B,J,xx0,aa0,aa1,N,m,l,nsteps);
            end proc;
            getwalk::static := proc(aa,h,nsteps)
                setvec(aa0,aa);
                walk(aa0,h,nsteps);
                return aa0;
            end proc;
            sample::static := proc(N,h,nsteps)
            local aa;
                B1 := g0:-sample(N);
                return rowmap(aa->getwalk(aa,h,nsteps),B1);
            end proc;
            getpath::static := proc(aa,h,nsteps)
                setvec(aa0,aa);
                n := walk(aa0,h,nsteps);
                return B[[seq(J[i],i=1..n)]];
            end proc;
            allocif::static := proc(nsteps)
                if(dyn:-allocif(nsteps)) then
                    B,J := dyn:-getelts();
                end if;
            end proc;
            init::static := proc()
                A,lm,beta := args;
                loss,lvars,lparams := op(lm);
                lossc := metlossc(loss,lvars,lparams)[1];
                walkc := bayeswalkc(loss,lvars,lparams);
                N,m := Dimension(A);
                l := nops(lparams);
                xx0,xx1,aa0,aa1 := allocla[float[8]](m,m,l,l);
                dyn := dynla(float[8](l),integer[4]);
                B,J := dyn:-getelts();
                g0 := gaussdist(l);
                g := gaussdist(l);
            end proc;
        end module;
        md:-init(A,lm,beta);
        return md;
    end proc;

    sampising0 := proc(G::Array(datatype=float[8]),A::Array(datatype=float[8]),V::Array(datatype=integer[4]),steps::integer[4],M::integer[4],N::integer[4],beta::float[8],init::integer[4]);
        if(init=1) then
            for i from 1 to N do
                for j from 1 to M do
                    if(rand() mod 2=0) then
                        A[i,j] := 1;
                    else
                        A[i,j] := -1;
                    end if;
                end do;
            end do;
        end if;
        for i from 1 to N do
            for k from 1 to steps do
                l := rand() mod M+1;
                a,b := 0.0,0.0;
                for j from 1 to M do
                    if(A[i,j]=-1) then
                        a := a+G[l,j];
                    else
                        b := b+G[l,j];
                    end if;
                end do;
                p,q := exp(-beta*a),exp(-beta*b);
                r := p/(p+q);
                r1 := evalf(rand() mod 10000000)/10000000.0;
                if(r1<r) then
                    A[i,l] := 1;
                else
                    A[i,l] := -1;
                end if;
            end do;
            H := 0.0;
            for l from 1 to M do
                for j from 1 to M do
                    if(A[i,l]<>A[i,j]) then
                        H := H+G[l,j];
                    end if;
                end do;
            end do;
            V[i] := round(H);
        end do;
    end proc;

    sampising0 := Compiler:-Compile(sampising0);

    sampising := proc(E,beta,steps,N)
        if(type(procname,indexed)) then
            Hl := [op(procname)];
            P,V := sampising(E,beta,steps,N);
            ans := [];
            for H in Hl do
                il := [];
                for i from 1 to N do
                    if(V[i]=H) then
                        il := [op(il),i];
                    end if;
                end do;
                ans := [op(ans),P[il,..]];
            end do;
            return op(ans);
        end if;
        G := gengraph(E);
        M := Dimension(G)[1];
        P := Matrix(N,M,datatype=float[8]);
        H := Vector(N,datatype=integer[4]);
        sampising0(G,P,H,steps,M,N,beta,1);
        return P,H;
    end proc;

    drawising := proc(V,G,col:='bw')
    uses GraphTheory;
        if(type(args[2],'Matrix')) then
            return procname(V,Graph(G));
        elif(not type(procname,indexed)) then
            return drawising[dimension=2,style='spring'](V,G);
        end if;
        cmap := colormap(col);
        M := Dimension(V);
        for i from 1 to M do
            if(V[i]>0) then
                c := [1,0,0];
            else
                c := [0,0,1];
            end if;
            c := cmap((V[i]+1)/2);
            HighlightVertex(G,i,c);
        end do;
        for e in Edges(G) do
            i,j := op(e);
            if(V[i]>=0 and V[j]>=0) then
                c := cmap(1);
            elif(V[i]>=0 and V[j]<0) then
                c := cmap(.5);
            elif(V[i]<0 and V[j]>=0) then
                c := cmap(.5);
            elif(V[i]<0 and V[j]<0) then
                c := cmap(0);
            end if;
            HighlightEdges(G,e,c);
        end do;
        return DrawGraph(G,op(procname));
    end proc;

    gengraph := proc(G)
        if(type(G,'Matrix')) then
            return tofloat8(G);
        elif(type(args[1],'function')) then
            F := args[1];
            if(op(0,F)='circ') then
                n := op(1,F);
                ans := Matrix(n,n,datatype=float);
                ans[1,2] := 1;
                ans[1,n] := 1;
                ans[n,1] := 1;
                ans[n,n-1] := 1;
                for i from 2 to n-1 do
                    ans[i,i-1] := 1;
                    ans[i,i+1] := 1;
                end do;
                return ans;
            elif(op(0,F)='interval') then
                n := op(1,F);
                ans := Matrix(n,n,datatype=float);
                ans[1,2] := 1;
                ans[n,n-1] := 1;
                for i from 2 to n-1 do
                    ans[i,i-1] := 1;
                    ans[i,i+1] := 1;
                end do;
                return ans;
            elif(op(0,F)='flares') then
                n := op(1,F);
                ans := Matrix(3*n-2,3*n-2,datatype=float);
                A1 := gengraph('interval'(n-1));
                ans[1..n-1,1..n-1] := A1;
                ans[n..2*n-2,n..2*n-2] := A1;
                ans[2*n-1..3*n-3,2*n-1..3*n-3] := A1;
                ans[n-1,3*n-2] := 1;
                ans[3*n-2,n-1] := 1;
                ans[2*n-2,3*n-2] := 1;
                ans[3*n-2,2*n-2] := 1;
                ans[3*n-3,3*n-2] := 1;
                ans[3*n-2,3*n-3] := 1;
                return ans;
            else
                error;
            end if;
        else
            error;
        end if;
    end proc;

end module;
