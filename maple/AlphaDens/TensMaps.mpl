TensMaps := module()
export allocarr,arrdim,arrtype,ordmaps,tensmaps,arr2vec,vec2arr,tensrng,vecpatches;
local tensord0,tensord1,tensmaps0,tensmaps1;

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

    arrtype := proc(arr)
        typl1 := ["Array","Vector","Matrix"];
        typl2 := [float[8],integer[4]];
        for typ1 in typl1 do
            if(type(arr,parse(typ1))) then
                for typ2 in typl2 do
                    if(type(arr,parse(cat(typ1,"(datatype=",typ2,")")))) then
                        return typ2;
                    end if;
                end do;
            end if;
        end do;
        error;
    end proc;

    tensrng := proc()
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
        return al,bl,ml,N;
    end proc;

    ordmap0 := proc()
        al,bl,ml,N := tensrng(args[1]);
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

    ordmap1 := proc()
        al,bl,ml,N := tensrng(args[1]);
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

    ordmaps := proc(rng)
        return ordmap0(rng),ordmap1(rng);
    end proc;

    tensmaps0 := proc(rng,dtype)
        al,bl,ml,N := tensrng(rng);
        d := nops(ml);
        code := "proc(vec::Array(datatype=";
        code := cat(code,convert(dtype,'string'));
        code := cat(code,"),arr::Array(datatype=");
        code := cat(code,convert(dtype,'string'));
        code := cat(code,"))\n");
        code := cat(code,"k := 0;\n");
        for j from 1 to d do
            a,b,m := al[j],bl[j],ml[j];
            code := cat(code,"for i",j," from ",a," to ",b," do\n");
        end do;
        code := cat(code,"k := k+1;\n");
        code := cat(code,"arr[");
        code := cat(code,"i1");
        for j from 2 to d do
            code := cat(code,",i",j);
        end do;
        code := cat(code,"] := vec[k];\n");
        for a from 1 to d do
            code := cat(code,"end do;\n");
        end do;
        code := cat(code,"end proc;\n");
        return Compiler:-Compile(parse(code));
    end proc;

    tensmaps1 := proc(rng,dtype)
        al,bl,ml,N := tensrng(rng);
        d := nops(ml);
        code := "proc(arr::Array(datatype=";
        code := cat(code,convert(dtype,'string'));
        code := cat(code,"),vec::Array(datatype=");
        code := cat(code,convert(dtype,'string'));
        code := cat(code,"))\n");
        code := cat(code,"k := 0;\n");
        for j from 1 to d do
            a,b,m := al[j],bl[j],ml[j];
            code := cat(code,"for i",j," from ",a," to ",b," do\n");
        end do;
        code := cat(code,"k := k+1;\n");
        code := cat(code,"vec[k] := arr[");
        code := cat(code,"i1");
        for j from 2 to d do
            code := cat(code,",i",j);
        end do;
        code := cat(code,"];\n");
        for a from 1 to d do
            code := cat(code,"end do;\n");
        end do;
        code := cat(code,"end proc;\n");
        return Compiler:-Compile(parse(code));
    end proc;

    tensmaps := proc(rng,dtype)
        return tensmaps0(rng,dtype),tensmaps1(rng,dtype);
    end proc;

    arr2vec0 := proc(d,dtype)
    option remember;
        code := "proc(arr::Array(datatype=";
        code := cat(code,convert(dtype,'string'));
        code := cat(code,"),vec::Array(datatype=");
        code := cat(code,convert(dtype,'string'));
        code := cat(code,")");
        for j from 1 to d do
            code := cat(code,",a",j,"::integer[4]");
        end do;
        for j from 1 to d do
            code := cat(code,",b",j,"::integer[4]");
        end do;
        code := cat(code,")\n");
        code := cat(code,"k := 0;\n");
        for j from 1 to d do
            code := cat(code,"for i",j," from a",j," to b",j," do\n");
        end do;
        code := cat(code,"k := k+1;\n");
        code := cat(code,"vec[k] := arr[");
        code := cat(code,"i1");
        for j from 2 to d do
            code := cat(code,",i",j);
        end do;
        code := cat(code,"];\n");
        for a from 1 to d do
            code := cat(code,"end do;\n");
        end do;
        code := cat(code,"end proc;\n");
        return Compiler:-Compile(parse(code));
    end proc;

    arr2vec := proc(arr,rng:=[arrdim(arr)])
    local vec;
        al,bl,ml,N := tensrng(rng);
        d := nops(al);
        typ := arrtype(arr);
        vec := allocla[typ](N);
        F := arr2vec0(d,typ);
        F(arr,vec,op(al),op(bl));
        return vec;
    end proc;

    vec2arr0 := proc(d,dtype)
    option remember;
        code := "proc(vec::Array(datatype=";
        code := cat(code,convert(dtype,'string'));
        code := cat(code,"),arr::Array(datatype=");
        code := cat(code,convert(dtype,'string'));
        code := cat(code,")");
        for j from 1 to d do
            code := cat(code,",a",j,"::integer[4]");
        end do;
        for j from 1 to d do
            code := cat(code,",b",j,"::integer[4]");
        end do;
        code := cat(code,")\n");
        code := cat(code,"k := 0;\n");
        for j from 1 to d do
            code := cat(code,"for i",j," from a",j," to b",j," do\n");
        end do;
        code := cat(code,"k := k+1;\n");
        code := cat(code,"arr[");
        code := cat(code,"i1");
        for j from 2 to d do
            code := cat(code,",i",j);
        end do;
        code := cat(code,"] := vec[k];\n");
        for a from 1 to d do
            code := cat(code,"end do;\n");
        end do;
        code := cat(code,"end proc;\n");
        return Compiler:-Compile(parse(code));
    end proc;

    vec2arr := proc(vec,rng)
        al,bl,ml,N := tensrng(rng);
        d := nops(al);
        typ := arrtype(vec);
        arr := allocarr[typ](ml);
        F := vec2arr0(d,typ);
        F(vec,arr,op(al),op(bl));
        return arr;
    end proc;

    vecpatches := proc(arr,l)
        ml := [arrdim(arr)];
        d := nops(ml);
        rng1 := [seq(l+1..ml[i]-l,i=1..d)];
        rng2 := [seq(-l..l,i=1..d)];
        f1,g1 := ordmaps(rng1);
        f2,g2 := ordmaps(rng2);
        N1,N2 := tensrng(rng1)[4],tensrng(rng2)[4];
        ans := matf(N1,N2);
        for i1 from 1 to N1 do
            xl1 := f1(i1);
            for i2 from 1 to N2 do
                xl2 := f2(i2);
                xl := xl1+xl2;
                ans[i1,i2] := arr[op(xl)];
            end do;
        end do;
        return ans;
    end proc;

    kronprod := proc()
        if(nargs=1) then
            return args[1];
        elif(nargs>2) then
            return kronprod(args[1..2],args[3..nargs]);
        end if;
        arr1,arr2 := args;
        ml1,ml2 := [arrdim(arr1)],[arrdim(arr2)];

    end proc;

end module;
