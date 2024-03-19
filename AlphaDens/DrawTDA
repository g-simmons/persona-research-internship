DrawTDA := module()
option package;
export drawplex,matplot,drawrange,plotarr,plotmats,projcoords,specplex,springplex,randframe,drawdims,drange,drawcloud,animpoints,heatplex;

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

    colorlist := proc(n)
        cols := ["black","green","red","blue","brown","purple","orange","pink","yellow"];
        if(n>nops(cols)) then
            error;
        end if;
        return cols[1..n];
    end proc;

    projcoords := proc(A,B)
        ans := A;
        if(nargs>1) then
            if(type(args[2],'numeric')) then
                return ans*args[2];
            end if;
            m1 := Dimension(B)[1];
            if(m1<m) then
                ans := ans[..,1..m1];
            end if;
            ans := ans.B;
        end if;
        return ans;
    end proc;

#random unitary frame of n vectors in R^m
    randframe := proc(n,m)
        B := matf(n,m);
        Sample(Normal(0,1),B);
        Q,R := QRDecomposition(B);
        return Q;
    end proc;

    plotmats := proc()
    uses plots,plottools;
        d := min(Dimension(args[1])[2],3);
        ans := [];
        cols := colorlist(nargs);
        for i from 1 to nargs do
            if(type(args[i],'Matrix')) then
                A := args[i];
                N := Dimension(A)[1];
                col := cols[i];
                ans := [op(ans),seq(point([seq(A[k,j],j=1..d)],symbol=solidcircle,symbolsize=5,color=col),k=1..N)];
            else
                A,cl := op(args[i]);
                N := Dimension(A);
                ans := [op(ans),seq(point([seq(A[k,j],j=1..d)],symbol=solidcircle,symbolsize=3,color=cl[k]),k=1..N)];
            end if;
        end do;
        if(type(procname,indexed)) then
            return display(ans,op(procname));
        else
            return display(ans);
        end if;
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

    drange := proc(A,flag:=false,h:=0.0)
        N,m := Dimension(A);
        rng := [];
        for j from 1 to m do
            a := min([seq(A[i,j],i=1..N)]);
            b := max([seq(A[i,j],i=1..N)]);
            rng := [op(rng),a-h..b+h];
        end do;
        if(flag) then
            r := max(seq(op(2,rng[j])-op(1,rng[j]),j=1..m))/2;
            for j from 1 to m do
                a,b := op(rng[j]);
                rng[j] := (a+b)/2-r..(a+b)/2+r;
            end do;
        end if;
        return rng;
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

#returns a list of suitable dimensions with maximum value of N
#for an image representing the given range
    drawdims := proc(N,rng)
        d := nops(rng);
        al := [seq(op(1,rng[i]),i=1..d)];
        bl := [seq(op(2,rng[i]),i=1..d)];
        if(type(N,'numeric')) then
            #t := min(seq(bl[i]-al[i],i=1..d))/N;
            t := max(seq(bl[i]-al[i],i=1..d))/N;
            ml := [seq(ceil((bl[i]-al[i])/t),i=1..d)];
            return ml;
        end if;
        ml := args[1];
        if(true in ml) then
            for i from 1 to d do
                if(type(ml[i],'numeric')) then
                    break;
                end if;
            end do;
            t := (bl[i]-al[i])/ml[i];
            return [seq(ceil((bl[i]-al[i])/t),i=1..d)];
        end if;
        return ml;
    end proc;

#draws an embedded image of the given complex in the coordinates of
#vmap. vmap is a vecmap object, or could be a matrix if the labels are
#the numbers 1..n.
    heatplex := proc(X,vmap,N,rng)
        if(type(vmap,'Matrix')) then
            return procname(X,vecmap(X:-getverts(),args[2]),args[3..nargs]);
        end if;
        if(nargs=2) then
            return procname(X,vmap,1000);
        elif(nargs=3) then
            S := vmap:-getmat();
            return heatplex(X,vmap,N,drange(S,false));
        end if;
        md := module()
        option object;
        export X,S,d,n,c0,c1,vmap,ml,rng,vertsize,edgesize,init,draw,cmap,sigl,al,bl,getcolor,getrgb,divcolor,fslack;
        local ModulePrint;
            ModulePrint::static := proc()
                draw();
                return nprintf("heatplex object, %d vertices",n);
            end proc;
            draw::static := proc()
            uses ImageTools,ImageTools:-Draw;
                img := Create(ml[2],ml[1],channels=3,background=white);
                for sig in sigl do
                    l := nops(sig);
                    pl := [seq(.98*vmap(x),x=sig)];
                    pl := [seq([seq((p[j]-al[j])/(bl[j]-al[j])*ml[j],j=1..d)],p=pl)];
                    c := getrgb(sig);
                    if(l=1) then
                        SolidCircle(img,op(pl[1]),vertsize,color='black');
                        SolidCircle(img,op(pl[1]),vertsize-2*edgesize,color=c);
                    elif(l=2) then
                        Line(img,op(pl[1]),op(pl[2]),color='black',thickness=edgesize);
                        #Line(img,op(pl[1]),op(pl[2]),color=c,thickness=edgesize);
                    elif(l=3) then
                        Poly(img,[pl[1],pl[2],pl[3],pl[1]],pattern="solid",color=c,fill_color=c,fill_pattern="solid");
                        Line(img,op(pl[1]),op(pl[2]),pattern="solid",color='black',thickness=edgesize);
                        Line(img,op(pl[1]),op(pl[3]),pattern="solid",color='black',thickness=edgesize);
                        Line(img,op(pl[2]),op(pl[3]),pattern="solid",color='black',thickness=edgesize);
                        SolidCircle(img,op(pl[1]),vertsize,color='black');
                        SolidCircle(img,op(pl[2]),vertsize,color='black');
                        SolidCircle(img,op(pl[3]),vertsize,color='black');
                    end if;
                end do;
                Embed(img);
                return img;
            end proc;
            divcolor::static := proc(vcols)
                for sig in sigl do
                    l := nops(sig);
                    a := X:-f(sig);
                    cl := add(vcols[sig[i]],i=1..l)/l;
                    t := 1-(a-c0)/(c1-c0);
                    getrgb(sig) := t*[1,1,1]+(1-t)*cl;
                end do;
            end proc;
            getrgb::static := proc(sig)
            option remember;
                return [getcolor(sig)[]];
            end proc;
            getcolor::static := proc(sig)
            option remember;
                a := X:-f(sig);
                return cmap((a-c0)/(c1-c0));
            end proc;
            init:= proc()
            local N;
                X,vmap,N,rng := args;
                d := 2;
                m := vmap:-m;
                n := X:-numverts();
                cl := [seq(X:-f(sig),sig=X[0])];
                c0,c1 := max(cl),min(cl);
                sigl := [seq(op(X[k]),k=[2,1,0])];
                ord := sort(map(X:-f,sigl),`>`,output=permutation);
                #ord := sort(map(fslack,sigl),`>`,output=permutation);
                sigl := sigl[ord];
                al := [seq(op(1,rng[i]),i=1..d)];
                bl := [seq(op(2,rng[i]),i=1..d)];
                S := vmap:-getmat();
                ml := drawdims(N,rng);
                vertsize := 8;
                edgesize := 2;
                cmap := colormap('grayblack');
            end proc;
            fslack::static := proc(sig)
                return X:-f(sig)+nops(sig)*.1*(c1-c0);
            end proc;
        end module;
        md:-init(args);
        return md;
    end proc;

    drawcloud := proc(A,rng,imsize)
    uses ImageTools,ImageTools:-Draw;
        if(not type(procname,indexed)) then
            return drawcloud['black',3](args);
        end if;
        col,pointsize := op(procname);
        m1,m2 := op(drawdims(imsize,rng));
        a1,b1 := op(rng[1]);
        a2,b2 := op(rng[2]);
        img := Create(m2,m1,channels=3,background=white);
        N,d := Dimension(A);
        for k from 1 to N do
            p := [(A[k,1]-a1)/(b1-a1)*m1,(A[k,2]-a2)/(b2-a2)*m2];
            SolidCircle(img,op(p),pointsize,color=col);
        end do;
        Embed(img);
        return img;
    end proc;

    floatcolor1 := proc(x)
        t := arctan(x)/Pi*2;
        p := min(max((t+1)/2,0.0),1.0);
        ans := [p,0,1-p];
        return ans;
    end proc;

    #draws an embedded image of the given complex in the coordinates of
#vmap. vmap is a vecmap object, or could be a matrix if the labels are
#the numbers 1..n.
drawplex := proc(X,vmap,N,rng)
    if(type(vmap,'Matrix')) then
        return procname(X,vecmap(X:-getverts(),args[2]),args[3..nargs]);
    end if;
    if(nargs=2) then
        return procname(X,vmap,1000);
    elif(nargs=3) then
        S := vmap:-getmat();
        return drawplex(X,vmap,N,drange(S,false));
    end if;
    md := module()
    option object;
    export X,S,d,n,c0,c1,vmap,ml,rng,vertsize,edgesize,init,draw,cmap,sigl,al,bl,getcolor,getrgb,divcolor,fslack;
    local ModulePrint;
        ModulePrint::static := proc()
            draw();
            return nprintf("drawplex object, %d vertices",n);
        end proc;
        draw::static := proc()
        uses ImageTools,ImageTools:-Draw;
            img := Create(ml[2],ml[1],channels=3,background=white);
            for sig in sigl do
                l := nops(sig);
                pl := [seq(.98*vmap(x),x=sig)];
                pl := [seq([seq((p[j]-al[j])/(bl[j]-al[j])*ml[j],j=1..d)],p=pl)];
                c := getrgb(sig);
                if(l=1) then
                    SolidCircle(img,op(pl[1]),vertsize,color='black');
                    SolidCircle(img,op(pl[1]),vertsize-2*edgesize,color=c);
                elif(l=2) then
                    Line(img,op(pl[1]),op(pl[2]),color='black',thickness=edgesize);
                    #Line(img,op(pl[1]),op(pl[2]),color=c,thickness=3*edgesize);
                elif(l=3) then
                    Poly(img,[pl[1],pl[2],pl[3],pl[1]],pattern="solid",color=c,fill_color=c,fill_pattern="solid");
                    Line(img,op(pl[1]),op(pl[2]),pattern="solid",color='black',thickness=edgesize);
                    Line(img,op(pl[1]),op(pl[3]),pattern="solid",color='black',thickness=edgesize);
                    Line(img,op(pl[2]),op(pl[3]),pattern="solid",color='black',thickness=edgesize);
                    SolidCircle(img,op(pl[1]),vertsize,color='black');
                    SolidCircle(img,op(pl[2]),vertsize,color='black');
                    SolidCircle(img,op(pl[3]),vertsize,color='black');
                end if;
            end do;
            Embed(img);
            return img;
        end proc;
        divcolor::static := proc(vcols)
            for sig in sigl do
                l := nops(sig);
                a := X:-f(sig);
                cl := add(vcols[sig[i]],i=1..l)/l;
                t := 1-(a-c0)/(c1-c0);
                getrgb(sig) := t*[1,1,1]+(1-t)*cl;
            end do;
        end proc;
        getrgb::static := proc(sig)
        option remember;
            return [getcolor(sig)[]];
            end proc;
        getcolor::static := proc(sig)
        option remember;
            a := X:-f(sig);
            return cmap((a-c0)/(c1-c0));
        end proc;
        init:= proc()
        local N;
            X,vmap,N,rng := args;
            d := 2;
            m := vmap:-m;
            n := X:-numverts();
            cl := [seq(X:-f(sig),sig=X[0])];
            c0,c1 := max(cl),min(cl);
            sigl := [seq(op(X[k]),k=[2,1,0])];
            ord := sort(map(X:-f,sigl),`>`,output=permutation);
            #ord := sort(map(fslack,sigl),`>`,output=permutation);
            sigl := sigl[ord];
            al := [seq(op(1,rng[i]),i=1..d)];
            bl := [seq(op(2,rng[i]),i=1..d)];
            S := vmap:-getmat();
            ml := drawdims(N,rng);
            vertsize := 8;
            edgesize := 2;
            cmap := colormap('viridis');
        end proc;
    end module;
    md:-init(args);
    return md;
end proc;

springplex := proc(X,dim)
uses GraphTheory;
    G := oneskel(X);
        if(dim=2) then
            out := [op(DrawGraph(G,style=spring,dimension=2))];
            out1 := [op(out[1])];
            ans := [];
            for A in out1 do
                if(type(A,'Matrix')) then
                    N,m := Dimension(A);
                    p := [seq(add(A[i,j],i=1..N)/N,j=1..m)];
                    ans := [op(ans),p];
                end if;
            end do;
            return matf(ans);
        elif(dim=3) then
            out := [op(DrawGraph(G,style=spring,dimension=3))];
            ans := [];
            for x in out do
                if(op(0,x)='MESH') then
                    arr := op(1,x);
                    N1,N2,m := arrdim(arr);
                    p := [seq(add(add(arr[i1,i2,j],i1=1..N1),i2=1..N2)/N1/N2,j=1..m)];
                    ans := [op(ans),p];
                end if;
            end do;
            return matf(ans);
        else
            error;
        end if;
    end proc;

    animpoints := proc(A1,A2,nsteps)
        N := Dimension(A1)[1];
        ans := [];
        for k from 1 to nsteps do
            t := (k-.5)/nsteps;
            ans1 := display([seq(point(t*[A2[i,1],A2[i,2]]+(1-t)*[A1[i,1],A1[i,2]],symbol=solidcircle,symbolsize=4,color=black),i=1..N)]);
            ans := [op(ans),ans1];
        end do;
        return display(op(ans),insequence=true);
    end proc;

end module;
