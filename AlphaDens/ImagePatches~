ImagePatches := module()
option package;
export im2vec,vec2im,drawpatch,dherm,tensvec,patchdata,imdim,allocim,drawim,loadmnist,getherm,toherm,kleinvec,map2im,ipcolmap,ind2tens,loadfmri,imdata,imdiff,snorm,sfilt;
local getcmpl;

    loadmnist := proc(typ)
        if(nargs=0) then
            return loadmnist('DIGTAB50');
        elif(typ='ALL') then
            return loadmnist0(typ);
        elif(typ='DIGTAB') then
            return loadmnist1(typ);
        elif(typ='DIGTAB50') then
            return loadmnist2(typ);
        else
            error;
        end if;
    end proc;

    loadmnist0 := proc()
        read(cat(datadir,"/mnist.csv"));
        il := convert(V1,'list');
        ans := table();
        for i from 0 to 9 do
            jl := findat(i,il);
            ans[i] := A1[jl,..]/255.0;
        end do;
        return ans;
    end proc;

    loadmnist1 := proc()
        read cat(datadir,"/mnist2");
        printf("use digtab[i] for i=0..9.\n");
    end proc;

    loadmnist2 := proc()
        read cat(datadir,"/mnist3");
        printf("just 50/digit. use digtab1[i].\n");
    end proc;

    loadfmri := proc()
        V1 := Import(cat(datadir,"/fmri/vectors/beep.csv"));
        V2 := convert(V1,'Array');
        V3 := Vector(V2[..,1],datatype=float[8]);
        A1 := vec2im[[91,109,91]](V3);
        return A1;
    end proc;

    vec2im0 := proc(V::Array(datatype=float[8]),C::Array(datatype=float[8]),m1::integer[4],m2::integer[4]);
        for i from 1 to m1 do
            for j from 1 to m2 do
                C[i,j] := V[(i-1)*m2+j];
            end do;
        end do;
    end proc;

    vec2im0 := Compiler:-Compile(vec2im0);

    vec2im1 := proc(V::Array(datatype=float[8]),C::Array(datatype=float[8]),m1::integer[4],m2::integer[4],m3::integer[4]);
        for i1 from 1 to m1 do
            for i2 from 1 to m2 do
                for i3 from 1 to m3 do
                    C[i1,i2,i3] := V[(i1-1)*m2*m3+(i2-1)*m3+i3];
                end do;
            end do;
        end do;
    end proc;

    vec2im1 := Compiler:-Compile(vec2im1);

    vec2im := proc(V)
        if(type(procname,indexed)) then
            return vec2im(args,op(procname));
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
            vec2im0(V,A,op(ml));
        elif(nops(ml)=3) then
            vec2im1(V,A,op(ml));
        else
            error;
        end if;
        return A;
    end proc;

    im2vec0 := proc(C::Array(datatype=float[8]),V::Array(datatype=float[8]),m1::integer[4],m2::integer[4]);
        for i from 1 to m1 do
            for j from 1 to m2 do
                V[(i-1)*m2+j] := C[i,j];
            end do;
        end do;
    end proc;

    im2vec0 := Compiler:-Compile(im2vec0);

    im2vec1 := proc(C::Array(datatype=float[8]),V::Array(datatype=float[8]),m1::integer[4],m2::integer[4],m3::integer[4]);
        for i from 1 to m1 do
            for j from 1 to m2 do
                for k from 1 to m3 do
                    V[(i-1)*m2*m3+(j-1)*m3+k] := C[i,j,k];
                end do;
            end do;
        end do;
    end proc;

    im2vec1 := Compiler:-Compile(im2vec1);

    im2vec := proc(A)
        if(type(procname,indexed)) then
            return im2vec(args,op(procname));
        end if;
        if(nargs=3) then
            V,ml := args[2..nargs];
        elif(nargs=2 and type(args[2],'list')) then
            ml := args[2];
            V := allocla(convert(ml,`*`));
        elif(nargs=2 and type(args[2],'Vector'(datatype=float[8]))) then
            V := args[2];
            ml := [imdim(A)];
        elif(nargs=1) then
            ml := [imdim(A)];
            V := allocla(convert(ml,`*`));
        else
            error;
        end if;
        if(nops(ml)=2) then
            im2vec0(A,V,op(ml));
        elif(nops(ml)=3) then
            im2vec1(A,V,op(ml));
        else
            error;
        end if;
        return V;
    end proc;

#object for storing images as arrays. converts the data to vector form
#using the array to coordinates conversions, and has points in space
#associated to each pixel.
    imdata := proc(ml)
        argl := [args];
        md := module()
        option object;
        export arr,getvec,setvec,cmap,ml,getrng,al,bl,d,N,draw,init,getdata,getpoint,getcoords,getind;
        local U1,V1;
            ModulePrint::static := proc()
                s := "%d";
                for j from 1 to d-1 do
                    s := cat(s,"x%d");
                end do;
                s := cat(s," image");
                return nprintf(s,op(ml));
            end proc;
            getvec::static := proc()
                U := opd[false=allocla(N),true=U1](procname,false);
                return im2vec[ml](arr,U);
            end proc;
            setvec::static := proc(U)
                vec2im[ml](U,arr);
                return;
            end proc;
            setmap::static := proc(f)
                for k from 1 to N do
                    il := getcoords(k);
                    V := getpoint[true](k);
                    for j from 1 to d do
                        c := f(V);
                        arr[op(il)] := c;
                    end do;
                end do;
                return;
            end proc;
            getdata::static := proc()
                return Matrix([seq(getpoint[](k),k=1..N)],datatype=float[8]);
            end proc;
            getpoint::static := proc(k)
                il := getcoords(k);
                ans := [seq(evalf(al[j]+(il[j]-.5)/ml[j]*(bl[j]-al[j])),j=1..d)];
                if(not type(procname,indexed) or op(procname)=false) then
                    V := allocla(d);
                elif(op(procname)=true) then
                    V := V1;
                elif(type(op(procname,'Vector'(datatype=float[8])))) then
                    V := op(procname);
                elif(nops(procname)=0) then
                    return ans;
                end if;
                for j from 1 to d do
                    V[j] := ans[j];
                end do;
                return V;
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
                return drawim[opd(procname,[])](arr,cmap);
            end proc;
            getrng::static := proc()
                return seq(al[i]..bl[i],i=1..d);
            end proc;
            init::static := proc()
                if(type(args[1],'list')) then
                    ml := args[1];
                    arr := allocim(ml);
                elif(type(args[1],'Array'(datatype=float[8]))) then
                    arr := args[1];
                    ml := [imdim(arr)];
                end if;
                d := nops(ml);
                N := convert(ml,`*`);
                V := allocla[float[8]](N);
                if(nargs=2) then
                    rng := args[2];
                else
                    rng := [seq(-1..1,i=1..d)];
                end if;
                al := [seq(evalf(op(1,r)),r=rng)];
                bl := [seq(evalf(op(2,r)),r=rng)];
                cmap := colormap('virdiv');
                return;
            end proc;
            init(op(argl));
        end module;
        return md;
    end proc;

    imdiff0 := proc(U1::Array(datatype=float[8]),a::integer[4],ml1::Array(datatype=integer[4]),d::integer[4],typ::integer[4])
        N := 1;
        for b from 1 to d do
            N := N*ml1[b];
        end do;
        t1 := 1;
        for b from a+1 to d do
            t1 := t1*ml1[b];
        end do;
        t2 := t1*ml1[a];
        for k from 1 to N do
            if(k-1 mod t2>=(ml1[a]-1)*t1) then
                next;
            end if;
            if(typ=0) then
                U1[k] := (U1[k]+U1[k+t1])/2;
            else
                U1[k] := U1[k+t1]-U1[k];
            end if;
        end do;
    end proc;

    imdiff0 := Compiler:-Compile(imdiff0);

    imdiff := proc(im,modes,s)
        l := nops(modes);
        ml1 := [imdim(im)];
        d := nops(ml1);
        ml2 := [seq(m-s,m=ml1)];
        if(min(ml2)<=0) then
            error;
        end if;
        im1 := allocim(ml1);
        mv := allocla[integer[4]](d);
        N1,N2 := convert(ml1,`*`),convert(ml2,`*`);
        U1,U2,A := allocla(N1,N2,[N2,l]);
        for j from 1 to l do
            p := modes[j];
            for a from 1 to d do
                mv[a] := ml1[a];
            end do;
            im2vec[ml1](im,U1);
            for a from 1 to d do
                for b from 1 to p[a] do
                    imdiff0(U1,a,mv,d,1);
                end do;
                for b from p[a]+1 to s do
                    imdiff0(U1,a,mv,d,0);
                end do;
            end do;
            vec2im[ml1](U1,im1);
            im2vec[ml2](im1,U2);
            c := evalf(1/mul(2^k/sqrt(ncr(s,k)),k=p));
            VectorScalarMultiply(U2,c,inplace);
            setcol(A,j,U2);
        end do;
        return A;
    end proc;

    snorm := proc(A,cl)
        N,m := Dimension(A);
        if(nargs=1) then
            return snorm(A,[seq(1,i=1..m)]);
        end if;
        return Vector([seq(sqrt(add(cl[j]*A[i,j]^2,j=1..m)),i=1..N)],datatype=float[8]);
    end proc;

    sfilt := proc(im,s,bl)
        ml,d := im:-ml,im:-d;
        k := nops(bl)-1;
        all := getcmpl(d,k);
        modes := all;
        cl := [seq(bl[convert(al,`+`)+1],al=all)];
        A := imdiff(im:-arr,modes,s);
        print(modes,cl);
        A1 := snorm(A,cl);
        ans := getim([seq(m-s,m=ml)]);
        ans:-rng := im:-rng;
        ans:-setvec(A1);
        return ans;
    end proc;

    drawpatch := proc(A)
        if(type(procname,indexed)) then
            return drawpatch(A,op(procname));
        end if;
        m1,m2 := Dimension(A);
        B := Matrix(10*m1,10*m2);
        for i1 from 1 to m1 do
            for i2 from 1 to m2 do
                for j1 from 2 to 9 do
                    for j2 from 2 to 9 do
                        B[(i1-1)*10+j1,(i2-1)*10+j2] := A[i1,i2];
                    end do;
                end do;
            end do;
        end do;
        if(nargs=2) then
            fn := args[2];
            ImageTools:-Write(fn,Create(B));
            return;
        end if;
        return Embed(Create(B));
    end proc;

    dherm := proc(ml,modes)
        if(not type(ml,'list')) then
            return dherm1(args);
        end if;
        l := nops(ml);
        if(l=1) then
            return dherm1(op(ml),map(op,modes));
        end if;
        m,il := ml[1],[seq(al[1],al=modes)];
        A1,A2 := dherm1(m,il);
        ml1,modes1 := ml[2..l],[seq(al[2..l],al=modes)];
        B1,B2 := dherm(ml1,modes1);
        C1,C2 := tensvec(A1,B1),tensvec(A2,B2);
        return C1,C2;
    end proc;

    dherm0 := proc(m)
        N := m-1;
        P,aa := allocla([N,m],N);
        k := 0;
        for i from 1 to N do
            k := k+1;
            P[k,i],P[k,i+1] := .5,.5;
            aa[k] := ncr(N-1,i-1)/2^(N-1);
        end do;
        return P,aa;
    end proc;

    dherm1 := proc(m,modes)
        if(type(modes,'integer')) then
            return dherm(m,0..modes);
        end if;
        P,aa := dherm0(m);
        H,B1,B2,La,L,bb := specdiff[true](P,aa);
        D1 := diag(rowsum(H));
        B1 := -B1[[seq(i+1,i=modes)],..];
        l,N := Dimension(B1);
        B2 := Matrix([seq([seq(B1[j,i]*ncr(N-1,i-1)/2^(N-1),i=1..N)],j=1..l)],datatype=float[8]);
        return B1,B2;
    end proc;

    tensvec0 := proc(U::Array(datatype=float[8]),V::Array(datatype=float[8]),W::Array(datatype=float[8]),m1::integer[4],m2::integer[4])
        for i1 from 1 to m1 do
            for i2 from 1 to m2 do
                W[(i1-1)*m2+i2] := U[i1]*V[i2];
            end do;
        end do;
    end proc;

    tensvec0 := Compiler:-Compile(tensvec0);

    tensvec1 := proc(A::Array(datatype=float[8]),B::Array(datatype=float[8]),C::Array(datatype=float[8]),N::integer[4],m1::integer[4],m2::integer[4])
        for i from 1 to N do
            for j1 from 1 to m1 do
                for j2 from 1 to m2 do
                    C[i,(j1-1)*m2+j2] := A[i,j1]*B[i,j2];
                end do;
            end do;
        end do;
    end proc;

    tensvec1 := Compiler:-Compile(tensvec1);

    tensvec := proc(U,V)
        if(nargs>2) then
            return tensvec(args[1],tensvec(args[2..nargs]));
        end if;
        if(type(U,'Vector')) then
            m,n := Dimension(U),Dimension(V);
            W := allocla(m*n);
            tensvec0(U,V,W,m,n);
            return W;
        elif(type(U,'Matrix')) then
            A,B := args[1..2];
            N,m := Dimension(A);
            n := Dimension(B)[2];
            C := allocla([N,m*n]);
            tensvec1(A,B,C,N,m,n);
            return C;
        else
            error;
        end if;
    end proc;

    getherm := proc(k,var)
    option remember;
    local x;
        if(nargs=1) then
            f := getherm(k,x);
            F := proc(y)
                return eval(f,x=y);
            end proc;
            return F;
        elif(type(k,'list')) then
            kl := k;
            d := nops(kl);
            return mul(getherm(kl[i],var[i]),i=1..d);
        end if;
        if(k=0) then
            return 1;
        end if;
        return simplify((-1)^k*exp(var^2/2)*diff(exp(-var^2/2),var$k))/sqrt(k!);
    end proc;

    toherm := proc(f,var)
        if(not type(procname,indexed)) then
            return toherm[H](args);
        end if;
        var1 := op(procname);
        ans := f;
        if(type(var,'list')) then
            varl := var;
        else
            varl := [var];
        end if;
        d := nops(varl);
        n := degree(f);
        all := getcmpl(d,n);
        ans := 0;
        for al in all do
            c := f;
            for i from 1 to d do
                c := coeff(c,varl[i],al[i]);
            end do;
            ans := ans+c*toherm1(al,var1);
        end do;
        return ans;
    end proc;

    toherm0 := proc(k,var)
    local x;
        A := Matrix(k+1,k+1);
        for i from 0 to k do
            f := getherm(i,x);
            for j from 0 to k do
                A[i+1,j+1] := coeff(f,x,j);
            end do;
        end do;
        B := A^(-1);
        return add(B[k+1,j+1]*var[j],j=0..k);
    end proc;

    toherm1 := proc(al,var)
        d := nops(al);
        n := pn(al);
        fl := [seq(toherm0(a,var),a=al)];
        bll := getcmpl(d,n);
        ans := 0;
        for bl in bll do
            ans := ans+mul(coeff(fl[i],var[bl[i]]),i=1..d)*var[op(bl)];
        end do;
        return ans;
    end proc;

    getcmpl := proc(n,k)
    option remember;
        if(k=0) then
            return [[seq(0,i=1..n)]];
        end if;
        all := getcmpl(n,k-1);
        ans := [];
        for al in all do
            ans := [op(ans),al];
            for i from 1 to n do
                bl := al;
                bl[i] := bl[i]+1;
                ans := [op(ans),bl];
            end do;
        end do;
        ans := [op({op(ans)})];
        sig := sort([seq(convert(al,`+`),al=ans)],output=permutation);
        return ans[sig];
    end proc;

    kleinvec := proc(tl,modes)
    local H,x,y;
        q1 := getherm(1);
        q2 := getherm(2);
        t1,t2 := op(tl);
        ans := sin(t2)*q1(cos(t1)*x+sin(t1)*y)+cos(t2)*q2(cos(t1)*x+sin(t1)*y);
        ans := toherm[H](ans,[x,y]);
        return Vector([seq(coeff(ans,H[op(p)]),p=modes)]);
    end proc;

    ipcolmap := proc(typ)
        if(nargs=0) then
            return ipcolmap('viridis');
        elif(type(typ,'object')) then
            return typ;
        end if;
        cm := colormap(typ);
        cm:-f := colpar('lin'(0..1));
        cm:-col0 := Color("white");
        cm:-col1 := Color("black");
        return cm;
    end proc;

    map2im := proc(f,rng)
        d := nops(rng);
        if(not type(op(1,procname),'list')) then
            return map2im[[op(procname)]](args);
        end if;
        ml := op(procname);
        al,bl := [seq(evalf(op(1,rng[i])),i=1..d)],[seq(evalf(op(2,rng[i])),i=1..d)];
        ans := allocim(ml);
        N := convert(ml,`*`);
        for k from 1 to N do
            il := ind2tens(k,ml);
            xl := [seq(al[j]+(il[j]-.5)*(bl[j]-al[j])/ml[j],j=1..d)];
            ans[op(il)] := evalf(f(op(xl)));
        end do;
        #return ans;
        im := imdata(ans,rng,args[3..nargs]);
        return im;
    end proc;

    ind2tens := proc(k,ml)
        d := nops(ml);
        a := k-1;
        ans := [];
        for i from d to 1 by -1 do
            r := a mod ml[i];
            ans := [r+1,op(ans)];
            a := (a-r)/ml[i];
        end do;
        return ans;
    end proc;

    drawim := proc(im,cm)
        if(type(procname,indexed) and nops(procname)<>0) then
            if(type(op(1,procname),'list')) then
                ml := op(procname);
            else
                ml := [op(procname)];
            end if;
        else
            ml := [200,200];
        end if;
        if(type(args[1],'procedure')) then
            return procname([args[1],args[2]],args[3..nargs]);
        elif(type(args[1],'list')) then
            im1 := map2im[ml](op(args[1]));
            return procname(im1,args[2..nargs]);
        elif(nargs=1) then
            return procname(im,ipcolmap('viridis'));
        end if;
        return imcmap(im,cm);
    end proc;

    patchdata1 := proc(im::Array(datatype=float[8]),A::Array(datatype=float[8]),s1::integer[4],d1::integer[4],l::integer[4],B::Array(datatype=float[8]),typ::integer[4],n::integer[4],V::Array(datatype=float[8]))
        N := d1;
        m := (2*s1+1);
        for i1 from 1 to d1 do
            i := i1;
            for j1 from -s1 to s1 do
                j := j1+s1+1;
                k1 := i1+j1;
                if(k1<1 or k1>d1) then
                    V[j] := 0.0;
                else
                    V[j] := im[k1];
                end if;
            end do;
            if(typ=0) then
                for j from 1 to m do
                    A[i,j] := V[j];
                end do;
            else
                for j from 1 to n do
                    c := 0.0;
                    for k from 1 to m do
                        c := c+B[k,j]*V[k];
                    end do;
                    A[i,j] := c;
                end do;
            end if;
        end do;
    end proc;

    patchdata1 := Compiler:-Compile(patchdata1);

    patchdata2 := proc(im::Array(datatype=float[8]),A::Array(datatype=float[8]),s1::integer[4],s2::integer[4],d1::integer[4],d2::integer[4],l::integer[4],B::Array(datatype=float[8]),typ::integer[4],n::integer[4],V::Array(datatype=float[8]))
        N := d1*d2;
        m := (2*s1+1)*(2*s2+1);
        for i1 from 1 to d1 do
            for i2 from 1 to d2 do
                i := (i1-1)*d2+i2;
                for j1 from -s1 to s1 do
                    for j2 from -s2 to s2 do
                        j := (j1+s1)*(2*s2+1)+j2+s2+1;
                        k1 := i1+j1;
                        k2 := i2+j2;
                        if(k1<1 or k1>d1 or k2<1 or k2>d2) then
                            V[j] := 0.0;
                        else
                            V[j] := im[k1,k2];
                        end if;
                    end do;
                end do;
                if(typ=0) then
                    for j from 1 to m do
                        A[i,j] := V[j];
                    end do;
                else
                    for j from 1 to n do
                        c := 0.0;
                        for k from 1 to m do
                            c := c+B[k,j]*V[k];
                        end do;
                        A[i,j] := c;
                    end do;
                end if;
            end do;
        end do;
    end proc;

    patchdata2 := Compiler:-Compile(patchdata2);

    patchdata3 := proc(im::Array(datatype=float[8]),A::Array(datatype=float[8]),s1::integer[4],s2::integer[4],s3::integer[4],d1::integer[4],d2::integer[4],d3::integer[4],l::integer[4],B::Array(datatype=float[8]),typ::integer[4],n::integer[4],V::Array(datatype=float[8]))
        N := d1*d2*d3;
        m := (2*s1+1)*(2*s2+1)*(2*s3+1);
        for i1 from 1 to d1 do
            for i2 from 1 to d2 do
                for i3 from 1 to d3 do
                    i := (i1-1)*d2*d3+(i2-1)*d3+i3;
                    for j1 from -s1 to s1 do
                        for j2 from -s2 to s2 do
                            for j3 from -s3 to s3 do
                                j := (j1+s1)*(2*s2+1)*(2*s3+1)+(j2+s2)*(2*s3+1)+j3+s3+1;
                                k1 := i1+j1;
                                k2 := i2+j2;
                                k3 := i3+j3;
                                if(k1<1 or k1>d1 or k2<1 or k2>d2 or k3<1 or k3>d3) then
                                    V[j] := 0.0;
                                else
                                    V[j] := im[k1,k2,k3];
                                end if;
                            end do;
                        end do;
                    end do;
                    if(typ=0) then
                        for j from 1 to m do
                            A[i,j] := V[j];
                        end do;
                    else
                        for j from 1 to n do
                            c := 0.0;
                            for k from 1 to m do
                                c := c+B[k,j]*V[k];
                            end do;
                            A[i,j] := c;
                        end do;
                    end if;
                end do;
            end do;
        end do;
    end proc;

    patchdata3 := Compiler:-Compile(patchdata3);

    patchdata := proc(im,sl)
        dl := [imdim(im)];
        l := nops(dl);
        if(type(args[2],'numeric')) then
            return patchdata(im,[seq(args[2],i=1..l)],args[3..nargs]);
        end if;
        ml := [seq(2*s+1,s=sl)];
        if(nargs=3 and type(args[3],'list')) then
            B2 := dherm(ml,args[3])[2];
            return patchdata(im,sl,Transpose(B2));
        end if;
        m := convert(ml,`*`);
        N := convert(dl,`*`);
        if(nargs=3) then
            B := args[3];
            n := Dimension(B)[2];
            typ := 1;
        else
            B := allocla([]);
            n := m;
            typ := 0;
        end if;
        A := allocla([N,n]);
        if(l=1) then
            patchdata1(im,A,sl[1],dl[1],l,B,typ,n,allocla(m));
        elif(l=2) then
            patchdata2(im,A,sl[1],sl[2],dl[1],dl[2],l,B,typ,n,allocla(m));
        elif(l=3) then
            patchdata3(im,A,sl[1],sl[2],sl[3],dl[1],dl[2],dl[3],l,B,typ,n,allocla(m));
        else
            error;
        end if;
        return A;
    end proc;

    imdim := proc(A)
        return seq(op(2,rng),rng=ArrayTools:-Dimensions(A));
    end proc;

    allocim := proc(ml)
        return Array(seq(1..m,m=ml),datatype=float[8]);
    end proc;

end module;
