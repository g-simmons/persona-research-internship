ImageMaps := module()
option package;
export map2arr,drawmap,getim,vec2arr,arr2vec,arrdim,allocarr,drawarr,colormap,colpar,map2im;
local colortab,colorvec,ModuleLoad;

    ModuleLoad := proc()
    global `type/ColorMap`,`type/ImageMap`,viewmode;
        `type/ColorMap` := proc(A)
            if(whattype(A)='ColorMap') then
                return true;
            else
                return false;
            end if;
        end proc;
        `type/ImageMap` := proc(A)
            if(whattype(A)='ImageMap') then
                return true;
            else
                return false;
            end if;
        end proc;
        viewmode := false;
    end proc;

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

    vec2arr0 := proc(V::Array(datatype=float[8]),C::Array(datatype=float[8]),m1::integer[4],m2::integer[4]);
        for i from 1 to m1 do
            for j from 1 to m2 do
                C[i,j] := V[(i-1)*m2+j];
            end do;
        end do;
    end proc;

    vec2arr0 := Compiler:-Compile(vec2arr0);

    vec2arr1 := proc(V::Array(datatype=float[8]),C::Array(datatype=float[8]),m1::integer[4],m2::integer[4],m3::integer[4]);
        for i1 from 1 to m1 do
            for i2 from 1 to m2 do
                for i3 from 1 to m3 do
                    C[i1,i2,i3] := V[(i1-1)*m2*m3+(i2-1)*m3+i3];
                end do;
            end do;
        end do;
    end proc;

    vec2arr1 := Compiler:-Compile(vec2arr1);

    vec2arr := proc(V)
        if(type(procname,indexed)) then
            return vec2arr(args,op(procname));
        elif(nargs=3) then
            A,ml := args[2..nargs];
        elif(type(args[2],'Array') or type(args[2],'Matrix')) then
            A := args[2];
            ml := [seq(op(2,rng),rng=ArrayTools:-Dimensions(A))];
        else
            ml := args[2];
            A := Array(seq(1..m,m=ml),datatype=float[8]);
        end if;
        if(nops(ml)=2) then
            vec2arr0(V,A,op(ml));
        elif(nops(ml)=3) then
            vec2arr1(V,A,op(ml));
        else
            error;
        end if;
        return A;
    end proc;

    arr2vec0 := proc(C::Array(datatype=float[8]),V::Array(datatype=float[8]),m1::integer[4],m2::integer[4]);
        for i from 1 to m1 do
            for j from 1 to m2 do
                V[(i-1)*m2+j] := C[i,j];
            end do;
        end do;
    end proc;

    arr2vec0 := Compiler:-Compile(arr2vec0);

    arr2vec1 := proc(C::Array(datatype=float[8]),V::Array(datatype=float[8]),m1::integer[4],m2::integer[4],m3::integer[4]);
        for i from 1 to m1 do
            for j from 1 to m2 do
                for k from 1 to m3 do
                    V[(i-1)*m2*m3+(j-1)*m3+k] := C[i,j,k];
                end do;
            end do;
        end do;
    end proc;

    arr2vec1 := Compiler:-Compile(arr2vec1);

    arr2vec := proc(A)
        if(type(procname,indexed)) then
            return arr2vec(args,op(procname));
        end if;
        if(nargs=3) then
            V,ml := args[2..nargs];
        elif(nargs=2 and type(args[2],'list')) then
            ml := args[2];
            V := allocla(convert(ml,`*`));
        elif(nargs=2 and type(args[2],'Vector'(datatype=float[8]))) then
            V := args[2];
            ml := [arrdim(A)];
        elif(nargs=1) then
            ml := [arrdim(A)];
            V := allocla(convert(ml,`*`));
        else
            error;
        end if;
        if(nops(ml)=2) then
            arr2vec0(A,V,op(ml));
        elif(nops(ml)=3) then
            arr2vec1(A,V,op(ml));
        else
            error;
        end if;
        return V;
    end proc;

    getball0 := proc(V::Array(datatype=float[8]),r::float[8],al::Array(datatype=float[8]),bl::Array(datatype=float[8]),ml::Array(datatype=integer[4]),epsl::Array(datatype=float[8]),inds::Array(datatype=integer[4]),d::integer[4],tl::Array(datatype=integer[4]),J0::Array(datatype=integer[4]),J1::Array(datatype=integer[4]))
        tl[d] := 1;
        N0 := 1;
        for j from 1 to d do
            eps := epsl[j];
            t := ceil(r/eps);
            tl[j] := t;
            N0 := N0*(2*t+1);
        end do;
        r2 := r*r;
        N1 := 0;
        for j from 1 to d do
            J1[j] := 0;
        end do;
        for j from 1 to d do
            J0[j] := ceil((V[j]-al[j])/(bl[j]-al[j])*ml[j]);
            end do;
        for i from 1 to N0 do
            for j from d-1 to 1 by -1 do
                j1 := j+1;
                t := 2*tl[j1]+1;
                if(J1[j1]>=t) then
                    J1[j1] := J1[j1]-t;
                    J1[j] := J1[j]+1;
                end if;
            end do;
            flag := 1;
            k := 0;
            d2 := 0.0;
            for j from 1 to d do
                a := J0[j]+J1[j]-tl[j];
                if(a<1 or a>ml[j]) then
                    flag := 0;
                    break;
                end if;
                k := k*ml[j]+(a-1);
                c := al[j]+(a-.5)/ml[j]*(bl[j]-al[j])-V[j];
                d2 := d2+c*c;
            end do;
            J1[d] := J1[d]+1;
            if(flag=0 or d2>=r2) then
                next;
            end if;
            N1 := N1+1;
            inds[N1] := k+1;
        end do;
        return N1;
    end proc;

    getball0 := Compiler:-Compile(getball0);

#object for storing images as arrays in space. converts the data to
#vector form using the array to coordinates conversions, and has
#points  in space associated to each pixel.
    getim := proc(ml,rng,cmap)
        argl := [args];
        md := module()
        option object;
        export arr,getvec,setvec,cmap,ml,getrng,al,bl,d,N,draw,init,getdata,getpoint,getcoords,getind,setarr,getcell,setmap,rng,epsl,getball,getball1,clear,`whattype`;
        local al1,bl1,ml1,epsl1,tl1,J0,J1,inds0,V1,ModuleApply;
            `whattype`::static := proc()
                return 'ImageMap';
            end proc;
            ModulePrint::static := proc()
                s := "%d";
                for j from 1 to d-1 do
                    s := cat(s,"x%d");
                end do;
                s := cat(s," image");
                return nprintf(s,op(ml));
            end proc;
            ModuleApply::static := proc(x)
                il := getcell(x);
                for k from 1 to d do
                    if(il[k]<1 or il[k]>ml[k]) then
                        return 0.0;
                    end if;
                end do;
                return arr[op(il)];
            end proc;
            clear::static := proc()
                ArrayTools:-Fill(0.0,arr);
            end proc;
            getvec::static := proc()
                if(not type(procname,indexed)) then
                    return getvec[allocla(N)](procname);
                end if;
                U := op(procname);
                return arr2vec[ml](arr,U);
            end proc;
            setvec::static := proc(U)
                vec2arr[ml](U,arr);
                return;
            end proc;
            setarr::static := proc(arr1)
                ArrayTools:-Copy(arr1,arr);
            end proc;
            setmap::static := proc(f)
                for k from 1 to N do
                    il := getcoords(k);
                    x := getpoint(k);
                    arr[op(il)] := f(x);
                end do;
                return;
            end proc;
            getball::static := proc(x,r)
                for j from 1 to d do
                    V1[j] := x[j];
                end do;
                N1 := getball1(V1,r,inds0);
                return [seq(inds0[i],i=1..N1)];
            end proc;
            getball1::static := proc(V,r,inds)
                return getball0(V,r,al1,bl1,ml1,epsl1,inds,d,tl1,J0,J1);
            end proc;
            getdata::static := proc()
                return Matrix([seq(getpoint(k),k=1..N)],datatype=float[8]);
            end proc;
            getpoint::static := proc(k)
                il := getcoords(k);
                ans := [seq(evalf(al[j]+(il[j]-.5)/ml[j]*(bl[j]-al[j])),j=1..d)];
                if(not type(procname,indexed)) then
                    return ans;
                elif(type(op(procname,'Vector'(datatype=float[8])))) then
                    V := op(procname);
                    for j from 1 to d do
                        V[j] := ans[j];
                    end do;
                    return V;
                else
                    error;
                end if;
            end proc;
            getcell::static := proc(x)
                ans := [seq(ceil((x[i]-al[i])/(bl[i]-al[i])*ml[i]),i=1..d)];
            end proc;
            getcoords::static := proc(k)
                k1 := k-1;
                ans := [];
                for j from d to 1 by -1 do
                    r := k1 mod ml[j];
                    ans := [r+1,op(ans)];
                    k1 := (k1-r)/ml[j];
                end do;
                return ans;
            end proc;
            getind::static := proc(il)
                ans := il[1]-1;
                for j from 2 to d do
                    ans := ans*ml[j]+il[j]-1;
                end do;
                ans := ans+1;
                return ans;
            end proc;
            draw::static := proc()
                if(type(procname,indexed)) then
                    return draw(args,op(procname));
                end if;
                if(nargs>0) then
                    return drawarr[args](arr,cmap);
                else
                    return drawarr(arr,cmap);
                end if;
            end proc;
            getrng::static := proc()
                return seq(al[i]..bl[i],i=1..d);
            end proc;
            init::static := proc()
                if(type(args[1],'list')) then
                    ml := args[1];
                    arr := allocarr(ml);
                elif(type(args[1],'Array'(datatype=float[8]))) then
                    arr := args[1];
                    ml := [arrdim(arr)];
                end if;
                d := nops(ml);
                N := convert(ml,`*`);
                if(nargs>=2) then
                    rng := args[2];
                else
                    rng := [seq(-1..1,i=1..d)];
                end if;
                if(nargs=3) then
                    cmap := args[3];
                else
                    cmap := colormap('virdiv');
                end if;
                al := [seq(evalf(op(1,r)),r=rng)];
                bl := [seq(evalf(op(2,r)),r=rng)];
                epsl := [seq((bl[j]-al[j])/ml[j],j=1..d)];
                al1 := Vector(al,datatype=float[8]);
                bl1 := Vector(bl,datatype=float[8]);
                ml1 := Vector(ml,datatype=integer[4]);
                epsl1 := Vector(epsl,datatype=float[8]);
                inds0,J0,J1,tl1 := allocla[integer[4]](N,d,d,d);
                V1 := allocla[float[8]](d);
                return;
            end proc;
            init(op(argl));
        end module;
        return md;
    end proc;

    map2arr := proc(f,rng)
        d := nops(rng);
        if(not type(op(1,procname),'list')) then
            return map2im[[op(procname)]](args);
        end if;
        ml := op(procname);
        al,bl := [seq(evalf(op(1,rng[i])),i=1..d)],[seq(evalf(op(2,rng[i])),i=1..d)];
        ans := allocarr(ml);
        N := convert(ml,`*`);
        for k from 1 to N do
            il := ind2tens(k,ml);
            xl := [seq(al[j]+(il[j]-.5)*(bl[j]-al[j])/ml[j],j=1..d)];
            ans[op(il)] := evalf(f(op(xl)));
        end do;
        im := getim(ans,rng,args[3..nargs]);
        return im;
    end proc;

    drawarr := proc(arr,cmap:='viridis')
    global viewmode;
        if(type(cmap,'symbol') or type(cmap,'string')) then
            return procname(arr,colormap(cmap));
        elif(type(arr,'Matrix'(datatype=float[8]))) then
            return procname(Array(arr,datatype=float[8]),cmap);
        elif(type(arr,'Matrix'(datatype=integer[4]))) then
            return procname(Array(arr,datatype=integer[4]),cmap);
        end if;
        m1,m2 := seq(op(2,rng),rng=ArrayTools:-Dimensions(arr));
        if(viewmode) then
            arr1 := Array(seq(1..m,m=[m2,m1,3]),datatype=float[8]);
        else
            arr1 := Array(seq(1..m,m=[m1,m2,3]),datatype=float[8]);
        end if;
        for i1 from 1 to m1 do
            for i2 from 1 to m2 do
                if(type(arr,'Array'(datatype=float[8]))) then
                    c := cmap(arr[i1,i2]);
                elif(type(arr,'Array'(datatype=integer[4]))) then
                    c := cmap[arr[i1,i2]];
                end if;
                if(viewmode) then
                    arr1[m2-i2+1,i1,1],arr1[m2-i2+1,i1,2],arr1[m2-i2+1,i1,3] := cmap(arr[i1,i2])[];
                else
                    arr1[i1,i2,1],arr1[i1,i2,2],arr1[i1,i2,3] := cmap(arr[i1,i2])[];
                end if;
            end do;
        end do;
        A := Create(arr1);
        if(type(procname,indexed) and nops(procname)<>0) then
            fn := op(procname);
            ImageTools:-Write(fn,A);
            return;
        end if;
        return Embed(A);
    end proc;

    drawmap := proc(f,rng,cmap)
        ml := op(procname);
        im := getim(ml,args[2..nargs]);
        im:-setmap(f);
        im:-draw(densfn);
    end proc;

end module;
