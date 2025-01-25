DrawTDA := module()
option package;
export colors,drawplex,specgraph,psvd,drawmap,drawvec,matplot,sgt,pallette,drawballs,drawrange,plotarr;

#spectral graph theory for a complex, or symmetric positive definite
#matrix, such as an adjacency matrix. the second parameter is a
#quadratic form so that as all the number of principal components
#approaches n, we obtain vertices for which C is the scalar
#products. the second matrix is similar, and can serve the role of the
#normalization factor of sqrt(D), indicated by setting to true.
    specgraph := proc(G,C)
        if(type(args[1],'FilteredComplex')) then
            K1 := args[1];
            labs := K1:-verts();
            n := nops(labs);
            G1 := dframe(labs,labs);
            for e in K1[1] do
                x,y := op(e);
                G1[x,y] := K1:-dv(e);
                G1[y,x] := G1[x,y];
            end do;
            return specgraph(G1,args[2..nargs]);
        elif(type(G,'DFrame')) then
            labs := G:-rlabs;
            argl := [G:-mat];
            if(nargs>=2) then
                if(type(args[2],'DFrame')) then
                    argl := [op(argl),args[2]:-sub(labs,labs):-mat];
                else
                    argl := [op(argl),args[2]];
                end if;
            end if;
            if(nargs>=3) then
                if(type(args[3],'DFrame')) then
                    argl := [op(argl),args[3]:-sub(labs):-mat];
                else
                    argl := [op(argl),args[3]];
                end if;
            end if;
            A := specgraph(op(argl));
            return dframe(A,labs);
        end if;
        n := Dimension(G)[1];
        if(nargs=1) then
            return specgraph(G,DiagonalMatrix([seq(1.0,i=1..n)],shape=diagonal,datatype=float));
        end if;
        L := Matrix(n,n,datatype=float);
        for i from 1 to n do
            for j from 1 to n do
                L[i,j] := -G[i,j];
            end do;
        end do;
        for i from 1 to n do
            L[i,i] := L[i,i]-add(L[i,j],j=1..n);
        end do;
        if(nargs=3 and args[3]<>false) then
            if(args[3]=true) then
                C2 := DiagonalMatrix([seq(1/max(1.0,add(G[i,j],j=1..n)-G[i,i]),i=1..n)],datatype=float,storage=diagonal);
            else
                C2 := args[3];
            end if;
            U2 := psvd(C2);
            L := U2.L.Transpose(U2);
        end if;
        U1 := psvd(C);
        C1 := Transpose(U1).L.U1;
        U0 := SingularValues(C1,output='U');
        #print(U1[1..3,1..3],U0[1..3,1..3]);
        ans := U1.U0;
        return ans;
    end proc;

#if not all singular values are positive, set them to zero. the output
#is U.sqrt(S).
    psvd := proc(C)
        n := Dimension(C)[1];
        S1 := SingularValues(C,output='S',datatype=float);
        U1 := SingularValues(C,output='U',datatype=float);
        for j from 1 to n do
            if(S1[j]<0) then
                for j1 from j to n do
                    for i from 1 to n do
                        U1[i,j1] := 0.0;
                    end do;
                end do;
                break;
            end if;
        end do;
        return U1.DiagonalMatrix([seq(sqrt(x),x=S1)],datatype=float,shape=diagonal);
    end proc;

    sgt := proc(E,d)
        if(nargs=2) then
            return sgt0(args);
        elif(nargs=3) then
            return sgt1(args);
        else
            error;
        end if;
    end proc;

    sgt0 := proc(E,d)
        M := Dimension(E)[1];
        V := Vector([seq(add(E[i,j],j=1..M),i=1..M)],datatype=float[8]);
        D2 := DiagonalMatrix(V,storage=diagonal);
        L := D2^(-1)-D2^(-1).E.D2^(-1);
        C1,S1,C2 := SingularValues(L,output=['U','S','Vt']);
        #C1,D1,C2 := svd(L);
        A := C1.DiagonalMatrix(S1);
        #M1 := Dimension(A)[2];
        return A[1..M,M-d..M-1];
    end proc;

    sgt1 := proc(E1,E2,d)
        if(type(E2,'list')) then
            return procname(E1,E1[..,args[2]],d);
        end if;
        M,n := Dimension(E2);
        print(hi,E1,M,n,d);
        A := sgt0(E1,d);
        print(hi,E2,M,n);
        D1 := DiagonalMatrix([seq(1/add(E2[i,j],i=1..M),j=1..n)],datatype=float[8]);
        print(E2,D1);
        B := E2.D1;
        return Transpose(B).A;
    end proc;

    drawvec := proc(V,col)
        if(nargs=1) then
            return procname(V,red);
        end if;
        N := Dimension(V);
        ans1 := [];
        for i from 1 to N-1 do
            ans1 := [op(ans1),line([i,V[i]],[i+1,V[i+1]],color=col)];
        end do;
        return display(ans1);
    end proc;

    plotarr := proc(arr,rng)
        if(nargs=2) then
            a,b := op(rng);
        else
            a,b := min(arr),max(arr);
        end if;
        if(type(procname,indexed)) then
            opts := [op(procname)];
        else
            opts := [];
        end if;
        if(type(arr,'Vector')) then
            N := Dimension(arr);
            ans := [];
            for i from 1 to N do
                x := (i-.5)/N;
                ans := [op(ans),point([x,arr[i]],symbol=solidcircle,op(opts))];
            end do;
        elif(type(arr,'Matrix')) then
            error "not implemented";
        else
            error;
        end if;
        return display(ans);
    end proc;

    drawmap := proc(f,r1,r2,norm:=true)
    uses ImageTools;
    local x,y;
        md := module()
        option object;
        export draw,m,n,f,r1,r2,norm,invert,img,tofile;
        local ModulePrint;
            draw::static := proc()
                if(type(f(0,0),`list`)) then
                    ans1 := drawmap0[m,n]((x,y)->f(x,y)[1],r1,r2,norm,true);
                    ans2 := drawmap0[m,n]((x,y)->f(x,y)[2],r1,r2,norm,true);
                    ans3 := drawmap0[m,n]((x,y)->f(x,y)[3],r1,r2,norm,true);
                    ans := CombineLayers(ans1,ans2,ans3);
                else
                    ans := drawmap0[m,n](f,r1,r2,norm,invert);
                end if;
                img := ans;
                return Embed(ans);
            end proc;
            tofile::static := proc(fn)
                ImageTools:-Write(fn,Create(img));
                return;
            end proc;
            ModulePrint::static := proc()
                draw();
                return nprintf("%dx%d drawmap object",m,n);
            end proc;
        end module;
        if(nargs=1) then
            return procname(f,0..1,0..1);
        elif(not type(procname,indexed)) then
            return procname[100,100](args);
        end if;
        m,n := op(procname);
        md:-f := f;
        md:-m := m;
        md:-n := n;
        md:-r1 := r1;
        md:-r2 := r2;
        md:-norm := norm;
        md:-invert := false;
        return md;
    end proc;

    drawmap0 := proc(f,r1,r2,norm,invert)
    uses ImageTools;
        m,n := op(procname);
        a,b := op(r1);
        c,d := op(r2);
        A := Matrix(m,n);
        for i from 1 to m do
            for j from 1 to n do
                x := a+(j-.5)/n*(b-a);
                y := c+(i-.5)/m*(d-c);
                A[m-i+1,j] := f(x,y);
            end do;
        end do;
        if(norm) then
            m1,m2 := min(A),max(A);
            r := m2-m1;
            if(r=0) then
                r := 1;
            end if;
            for i from 1 to m do
                for j from 1 to n do
                    A[i,j] := (A[i,j]-m1)/r;
                end do;
            end do;
        end if;
        if(not invert) then
            for i from 1 to m do
                for j from 1 to n do
                    A[i,j] := 1-A[i,j];
                end do;
            end do;
        end if;
        return Create(A);
    end proc;

    matplot := proc(Al,cols)
        if(not type(args[1],'list')) then
            return procname([args]);
        end if;
        l := nops(Al);
        if(nargs=1) then
            cols0 := ["blue","red","green","black","purple"];
            if(l>nops(cols0)) then
                cols0 := [op(cols0),seq("black",i=nops(cols0)+1..l)];
            end if;
            return procname(Al,cols0);
        end if;
        md := module()
        option object;
        export draw,Al,dim,cols,dens,l,symbsize,getcol,getcol0,getdens,getview,h,mindens,pcaref,ell,X;
        local ModulePrint,ModuleApply;
            ModulePrint::static := proc()
                return draw();
                #s := "%d-dimensional, matrix plot, %d matrices";
                #return nprintf(s,dim,l);
            end proc;
            ModuleApply::static := proc()
                return draw();
            end proc;
            getcol::static := proc(k,i)
                c := cols[k];
                if(type(c,'Vector') or type(c,'list')) then
                    c := c[i];
                end if;
                ans := getcol0(c);
                d := getdens(k,i);
                cl := [ans[]];
                return Color(cl*d+(1-d)*[1.0,1.0,1.0]);
            end proc;
            getcol0::static := proc(c)
                if(type(c,'float')) then
                    p := min(max((c+1)/2,0.0),1.0);
                    if(p<.5) then
                        p := 2*p;
                        ans := [p,0,1-p];
                    else
                        p := 2*p-1;
                        ans := [1,p,0];
                    end if;
                    return Color(ans);
                else
                    return Color(c);
                end if;
            end proc;
            getdens::static := proc(k,i)
                return dens[k][i];
            end proc;
            draw::static := proc()
                Nl := [seq(Dimension(A)[1],A=Al)];
                if(pcaref=[]) then
                    A1 := Matrix([seq([A],A=Al)],datatype=float[8]);
                    rho1 := Vector([seq(rho,rho=dens)],datatype=float[8]);
                    pcaref1 := [A1,rho1];
                else
                    pcaref1 := pcaref;
                end if;
                if(dim<>Dimension(Al[1])[2]) then
                    Bl := mpca(Al,op(pcaref1));
                else
                    Bl := Al;
                end if;
                ans := [];
                for k from 1 to l do
                    B := Bl[k];
                    cl := cols[k];
                    el := ell[k];
                    for e in [indices(el)] do
                        i1,i2 := op(e);
                        d := min(getdens(k,i1),getdens(k,i2));
                        if(d<mindens) then
                            next;
                        end if;
                        d1 := ceil(symbsize*d);
                        col1 := [getcol(k,i1)[]];
                        col2 := [getcol(k,i2)[]];
                        col := Color((col1+col2)/2);
                        p := [seq(B[i1,j],j=1..dim)];
                        q := [seq(B[i2,j],j=1..dim)];
                        t := el[i1,i2];
                        ans := [op(ans),line(p,q,color=col,thickness=d1,transparency=1-t)];
                    end do;
                end do;
                for k from 1 to l do
                    B := Bl[k];
                    for i from 1 to Nl[k] do
                        d := getdens(k,i);
                        if(d<mindens) then
                            next;
                        end if;
                        col := getcol(k,i);
                        p := [seq(B[i,j],j=1..dim)];
                        t := ceil(symbsize*d);
                        ans := [op(ans),point(p,color=col,symbolsize=t,symbol='solidcircle')];
                    end do;
                end do;
                if(dim=3) then
                    return display(ans,view=getview(Bl),lightmodel='none');
                else
                    return display(ans,view=getview(Bl));
                end if;
            end proc;
            getview::static := proc(Bl)
                al := [seq(infinity,i=1..dim)];
                bl := [seq(-infinity,i=1..dim)];
                for i from 1 to l do
                    rho := dens[i];
                    B := Bl[i];
                    V,rl := meanstd(B,rho);
                    al := [seq(min(al[i],V[i]-h*rl[i]),i=1..dim)];
                    bl := [seq(max(bl[i],V[i]+h*rl[i]),i=1..dim)];
                end do;
                r := max(seq(bl[i]-al[i],i=1..dim));
                cl := [seq((al[i]+bl[i])/2,i=1..dim)];
                ans := [seq(cl[i]-r..cl[i]+r,i=1..dim)];
                return ans;
            end proc;
        end module;
        md:-mindens := .1;
        md:-h := 1.0;
        md:-symbsize := 5;
        md:-l := nops(Al);
        md:-Al := Al;
        md:-ell := [seq(table(),i=1..l)];
        md:-dim := min(3,Dimension(Al[1])[2]);
        md:-cols := cols;
        md:-dens := [seq(Vector([seq(1.0,i=1..Dimension(A)[1])],datatype=float[8]),A=Al)];
        md:-pcaref := [];
        return md;
    end proc;

    drawrange := proc(A,h:=0.0)
        if(type(A,'list')) then
            A1,dim := op(A);
        else
            A1 := A;
            dim := Dimension(A1)[2];
        end if;
        n := Dimension(A1)[1];
        rngl := [];
        for j from 1 to dim do
            a := min([seq(A1[i,j],i=1..n)]);
            b := max([seq(A1[i,j],i=1..n)]);
            rngl := [op(rngl),a..b];
        end do;
        c := 0.0;
        for i from 1 to dim do
            rng := rngl[i];
            a,b := op(rng);
            c := max(c,(b-a)/2);
        end do;
        c := c+h;
        ans := [];
        for i from 1 to dim do
            a,b := op(rngl[i]);
            x := (a+b)/2;
            ans := [op(ans),x-c..x+c];
        end do;
        return op(ans);
    end proc;

    drawplex := proc(A,X)
        if(nargs=1) then
            n := Dimension(A)[1];
            X1 := plex();
            for i from 1 to n do
                X1:-addsimps([i]=0.0);
            end do;
            return drawplex(A,X1);
        end if;
        argl := [A,X];
        md := module()
        option object;
        export A,X,n,m,dim,draw,vertsizes,symbsize,setsizes,setcolor,getcolor,getdens,getsimps,getview,h,minsize,mindens,coords,drawlabs,cols,pal,dens,C,d1,d2;
        local ModulePrint;
            ModulePrint::static := proc()
                draw();
            end proc;
            setcolor::static := proc(V)
                for i from 1 to n do
                    if(type(V,'list') or type(V,'Vector')) then
                        cols := pal(V[i]);
                    else
                        cols := pal(V);
                    end if;
                    for j from 1 to 3 do
                        C[i,j] := cols[j];
                    end do;
                end do;
                return;
            end proc;
            getcolor::static := proc()
                if(type(args[1],'numeric')) then
                    i := args[1];
                    ans := [C[i,1],C[i,2],C[i,3]];
                    d := getdens([i]);
                elif(type(args[1],'list')) then
                    il := args[1];
                    l := nops(il);
                    ans := add([C[i,1],C[i,2],C[i,3]],i=il)/l;
                    d := getdens(il);
                else
                    error;
                end if;
                ans := d*ans+(1-d)*[1.0,1.0,1.0];
                return Color(ans);
            end proc;
            setsizes::static := proc(sizes,normflag:=true)
                if(normflag) then
                    c := max(sizes);
                else
                    c := 1.0;
                end if;
                for i from 1 to n do
                    vertsizes[i] := sizes[i]/c;
                end do;
                return;
            end proc;
            getdens::static := proc(sig)
                l := nops(sig);
                return dens[l][sig];
            end proc;
            draw::static := proc()
                if(nargs=1) then
                    coords := args[1];
                end if;
                ans0 := [];
                ans1 := [];
                ans2 := [];
                for sig in getsimps() do
                    l := nops(sig);
                    d := getdens(sig);
                    s := min(seq(vertsizes[i],i=sig));
                    if(s<minsize or d<mindens) then
                        next;
                    end if;
                    s1 := ceil(symbsize*s);
                    col := getcolor(sig);
                    pl := [seq([seq(A[i,j],j=coords)],i=sig)];
                    if(l=1) then
                        ans0 := [op(ans0),point(pl[1],color=col,symbolsize=s1,symbol='solidcircle')];
                    elif(l=2) then
                        ans1 := [op(ans1),line(pl[1],pl[2],color=col,thickness=s1)];
                    elif(l=3) then
                        ans2 := [op(ans2),polygon(pl,linestyle=none,color=col)];
                    end if;
                end do;
                ans2 := revl(ans2);
                ans := [op(ans2),op(ans1),op(ans0)];
                if(nops(coords)=3) then
                    return display(ans,view=getview(A),lightmodel='none');
                else
                    return display(ans,view=getview(A));
                end if;
            end proc;
            getsimps::static := proc(k)
            option remember;
                sigl := X[args];
                sig := sort(map(X,sigl),`>`,output=permutation);
                sigl := sigl[sig];
                return sigl;
            end proc;
            getview::static := proc(A)
                d := nops(coords);
                al := [seq(infinity,i=1..d)];
                bl := [seq(-infinity,i=1..d)];
                V,rl := meanstd(A,vertsizes);
                al := [seq(min(al[i],V[i]-h*rl[i]),i=1..d)];
                bl := [seq(max(bl[i],V[i]+h*rl[i]),i=1..d)];
                r := max(seq(bl[i]-al[i],i=1..d));
                cl := [seq((al[i]+bl[i])/2,i=1..d)];
                ans := [seq(cl[i]-r..cl[i]+r,i=1..d)];
                return ans;
            end proc;
            init::static := proc(A,X)
                thismodule:-X,thismodule:-A := X,A;
                verts := X:-verts();
                n,m := Dimension(A);
                if(n<>nops(verts)) then
                    error "matrix should match number of vertices";
                end if;
                minsize := .1;
                mindens := 0.0;
                h := 1.0;
                symbsize := 5;
                d := min(m,2);
                coords := [seq(i,i=1..d)];
                drawlabs := false;
                cols := Vector(n);
                pal := pallette('REDBLUE');
                C := Matrix(n,3,datatype=float[8]);
                setcolor("black");
                dens := [];
                d1 := exp(-min(seq(X([i]),i=verts)))*1.0;
                d2 := 0.0;
                for k from 0 to 2 do
                    dtab := table();
                    for sig in X[k] do
                        dtab[sig] := (min(d1,exp(-X(sig)))-d2)/(d1-d2);
                    end do;
                    dens := [op(dens),eval(dtab)];
                end do;
                vertsizes := Vector([seq(1.0,i=1..n)],datatype=float[8]);
                dim := 2;
            end proc;
            init(op(argl))
        end module;
        return md;
    end proc;

    symbcolor0 := proc()
        c1,c2,c3 := randf(0,1),randf(0,1),randf(0,1);
        c := c1+c2+c3;
        return [c1,c2,c3]/c;
    end proc;

    symbcolor1 := proc()
        p := randf(0,1);
        ans := [p,0,1-p];
    end proc;

    floatcolor0 := proc(x)
        t := arctan(x)/Pi*2;
        p := min(max((t+1)/2,0.0),1.0);
        if(p<=.2) then
            p := 5*p;
            p := (p+1)/2;
            ans := [0,(1-p),p];
        elif(p>=.8) then
            p := 5-5*p;
            p := (p+1)/2;
            ans := [1,(1-p),0];
        else
            p := (p-.2)/.6;
            ans := [p,0,1-p];
        end if;
        return ans;
    end proc;

    floatcolor1 := proc(x)
        t := arctan(x)/Pi*2;
        p := min(max((t+1)/2,0.0),1.0);
        ans := [p,0,1-p];
        return ans;
    end proc;

    pallette := proc(s)
        md := module()
        option object;
        export s,getcolor,h,floatcolor,symbcolor,refresh;
        local ModuleApply,ModulePrint;
            ModulePrint::static := proc()
                s0 := "color pallette, type=";
                return nprintf(cat(s0,convert(s,'string')));
            end proc;
            ModuleApply::static := proc()
                return getcolor(args);
            end proc;
            getcolor::static := proc(x)
                if(type(x,'float')) then
                    return Color(floatcolor(x/h));
                elif(type(x,'list') and nops(x)=3 and type(x[1],'float')) then
                    return Color(x);
                elif(type(x,'string')) then
                    return Color(x);
                else
                    return Color(symbcolor(args));
                end if;
            end proc;
            refresh::static := proc()
                forget(floatcolor);
                forget(symbcolor);
            end proc;
        end module;
        md:-h := 1.0;
        md:-s := s;
        if(s='HOTCOLD') then
            md:-floatcolor := remem(floatcolor0);
            md:-symbcolor := remem(symbcolor0);
        elif(s='REDBLUE') then
            md:-floatcolor := remem(floatcolor1);
            md:-symbcolor := remem(symbcolor1);
        else
            error;
        end if;
        return md;
    end proc;

end module;
