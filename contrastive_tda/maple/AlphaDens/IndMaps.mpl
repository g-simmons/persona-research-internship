TensMaps := module()
export allocarr,arrdim,tensord;

    arrdim := proc(A)
        return seq(op(2,rng),rng=ArrayTools:-Dimensions(A));
    end proc;

    allocarr := proc()
        if(not type(procname,indexed)) then
            return allocarr[float[8]](args);
        end if;
        typ := op(procname);
        ans := [];
        for ml in args do
            ans := [op(ans),Array(seq(1..m,m=ml),datatype=typ)];
        end do;
        return op(ans);
    end proc;

    tensord0 := proc(al,ml)
        d := nops(al);
        code := "proc(j)\n";
        code := cat(code,"option inline;\n");
        code := cat(code,"return [");
        for i from 1 to d-1 do
            m1 := mul(ml[j],j=i..d);
            m2 := mul(ml[j],j=i+1..d);
            code := cat(code,al[i],"+iquo(irem(j-1,",m1,"),",m2,"),");
        end do;
        code := cat(code,al[d],"+irem(j-1,",ml[d],")];\n");
        code := cat(code,"end proc;\n");
        return parse(code);
    end proc;

    tensord1 := proc(al,ml)
        d := nops(al);
        cl := [seq(mul(ml[j],j=i+1..d),i=1..d)];
        c0 := 1-add(al[i]*cl[i],i=1..d);
        code := "proc(xl)\n";
        code := cat(code,"option inline;\n");
        code := cat(code,"return ",c0);
        for i from 1 to d do
            code := cat(code,"+",cl[i],"*xl[",i,"]");
        end do;
        code := cat(code,";\n");
        code := cat(code,"end proc;\n");
        return parse(code);
    end proc;

    tensord := proc(rng)
        argl := [rng];
        md := module()
        option object;
        export d,N,getrng,getdims,getelt,getelts,getind,getinds,getsize,init,`numelems`,`?[]`;
        local ModulePrint,al,bl,ml;
            ModulePrint::static := proc()
                ans := "c-ordering of [%d..%d";
                ans := cat(ans,seq(",%d..%d",i=1..d-1));
                ans := cat(ans,"]");
                return nprintf(ans,seq(op(r),r=rng));
            end proc;
            getsize::static := proc()
                return N;
            end proc;
            `numelems`::static := getsize;
            init::static := proc()
            local rng;
                rng0 := args[1];
                rng := [];
                for r in rng0 do
                    if(type(r,'numeric')) then
                        rng := [op(rng),1..r];
                    else
                        rng := [op(rng),r];
                    end if;
                end do;
                d := nops(rng);
                al := [seq(op(1,r),r=rng)];
                bl := [seq(op(2,r),r=rng)];
                ml := [seq(bl[i]-al[i]+1,i=1..d)];
                N := convert(ml,`*`);
            end proc;
            init(op(argl));
            getrng::static := proc()
                return rng;
            end proc;
            getdims::static := proc()
                return ml;
            end proc;
            getelt::static := tensord0(al,ml);
            getelts::static := proc(inds:=1..N)
                return Matrix([seq(getelt(i),i=inds)],datatype=integer[4]);
            end proc;
            `?[]`::static := proc()
                inds := op(args[2]);
                print(inds);
                if(inds=..) then
                    inds := 1..N;
                end if;
                return getelts(op(args[2]));
            end proc;
            getind::static := tensord1(al,ml);
            getinds::static := proc(E)
                N1 := Dimension(E)[1];
                return Vector([seq(getind([seq(E[i,j],j=1..d)]),i=1..N1)],datatype=integer[4]);
            end proc;
        end module;
        md:-init(rng);
        return md;
    end proc;

end module;
