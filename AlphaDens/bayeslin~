#generic mixture of relationships with noise
genmix := proc(fl,rng,h,N)
    if(type(procname,indexed)) then
        var := op(procname);
    else
        var := x;
    end if;
    l := nops(fl);
    a,b := op(rng);
    ans := matf(l*N,2);
    V := Sample(Normal(0,h),l*N);
    k := 0;
    for f in fl do
        for i from 1 to N do
            x1 := randf(a,b);
            y1 := eval(f,var=x1);
            k := k+1;
            y1 := y1+V[k];
            ans[k,1],ans[k,2] := x1,y1;
        end do;
    end do;
    return ans;
end proc;

lincut := proc(A,yy,c,beta,h)
    md := module()
    option object;
    export A,yy,N,m,l,beta,h,walk,getwalk,sample,initstate,plotdiff,plotpath,diffvec,diffcloud,init;
    local ModulePrint,aa0,yy0;
        ModulePrint::static := proc()
            return nprintf("linear model with cutoff loss, %d points",N);
        end proc;
        initstate::static := proc()
            for j from 1 to m do
                aa0[j] := 0.0;
            end do;
            aa0[l] := randelt(yy);
            return aa0;
        end proc;
        sample::static := proc(N1,nsteps)
        local aa;
            B := rowmap(initstate,N1);
            ans := rowmap(aa->getwalk(aa,nsteps),B);
            return ans;
        end proc;
        plotpath::static := proc()
            return Metropolis:-plotpath(initstate());
        end proc;
        diffvec::static := proc(aa)
            yy1 := walk:-predvec(aa);
            for i from 1 to N do
                a := yy1[i]-yy[i];
                a := max(min(a,c),-c);
                yy0[i] := a;
            end do;
            return yy0;
        end proc;
        plotdiff::static := proc(aa)
            yy1 := diffvec(aa);
            return display([seq(point([yy[i],yy1[i]],symbol=solidcircle,color=black,symbolsize=3),i=1..N)]);
        end proc;
        diffcloud::static := proc(N1,nsteps)
            A1 := sample(N1,nsteps);
            ans := rowmap(diffvec,A1);
            if(nargs=3) then
                M := args[3];
                R := PCA(ans);
                ans := R:-principalcomponents[..,1..M];
            end if;
            return ans;
        end proc;
        init::static := proc()
        local x,a,b,c;
            A,yy,c1,beta,h := args;
            N,m := Dimension(A);
            l := m+1;
            pred := [a0+add(cat(a,i)*cat(x,m),i=1..m),[seq(cat(x,i),i=1..m)],[seq(cat(a,i),i=0..m)]];
            loss := [min(x^2,c^2),x,[c=c1]];
            walk := bayeswalk(A,yy,pred,loss,beta,h);
            getwalk := walk:-getwalk;
            yy0,aa0 := allocla[float[8]](N,l);
        end proc;
    end module;
    md:-init(A,yy,c,beta,h);
    return md;
end proc;
