DrawAlpha := module()
option package;
export drawplex,drawalpha,heatshape,rastrange,powheat,endrange,powrange;
local pnghack,topng;

    pnghack := proc(m,n)
        im := Array(1..m,1..n,1..3,datatype=float[8]);
        ArrayTools:-Fill(1.01,im);
        return im;
    end proc;

    topng0 := proc(im1::Array(datatype=float[8]),im2::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for i from 1 to m do
            for j from 1 to n do
                if(im1[i,j,1]>1) then
                    im2[i,j,4] := 1.0;
                else
                    for k from 1 to 3 do
                        im2[i,j,k] := im1[i,j,k];
                    end do;
                end if;
            end do;
        end do;
    end proc;

    topng0 := Compiler:-Compile(topng0);

    topng := proc(im)
        m,n,d := arrdim(im);
        ans := allocarr[float[8]]([m,n,4]);
        topng0(im,ans,m,n);
        return ans;
    end proc;

    drawplex := module()
    option object;
    export setplex,draw,getcolor,setcolor,cmap,vertsize,edgesize,sigl,cl,al,bl,vmap,S,c0,c1,rng,d,N,channels,`?[]`;
    local ModulePrint,ModuleApply;
        ModulePrint::static := proc()
            return nprintf("draws an alpha complex");
        end proc;
        setplex::static := proc(X,S,a1:=true,r:=1.02)
            if(type(S,'Matrix')) then
                thismodule:-S := S;
                vmap := vecmap(X:-getverts(),S);
            else
                vmap := S;
                thismodule:-S := vmap:-getmat();
            end if;
            sigl := [seq(op(X[k]),k=[2,1,0])];
            cl := [seq(X:-f(sig),sig=sigl)];
            N := nops(sigl);
            ord := sort(cl,`>`,output=permutation);
            sigl,cl := sigl[ord],cl[ord];
            if(a1=true) then
                c0,c1 := min(cl),max(cl);
            elif(type(a1,'numeric')) then
                c0,c1 := min(cl),a1;
            else
                c0,c1 := op(a1);
            end if;
            if(not type(r,'numeric')) then
                rng := r;
                al,bl := endrange(rng);
            else
                al,bl := endrange(drange(S));
                d := nops(al);
                for i from 1 to d do
                    al[i] := al[i]-r;
                    bl[i] := bl[i]+r;
                end do;
                rng := [seq(al[i]..bl[i],i=1..d)];
            end if;
            d := nops(rng);
            return;
        end proc;
        draw::static := proc()
        uses ImageTools,ImageTools:-Draw;
            if(type(procname,indexed)) then
                print(hi1,args);
                setplex(args);
                return draw(op(procname));
            elif(d<2) then
                error;
            elif(nargs=0) then
                return draw(1000);
            end if;
            if(type(M,'Array')) then
                im := M;
                ml,dxl := rastrange(rng,arrdim(im));
            else
                ml,dxl := rastrange(rng,args[1..nargs]);
                im := pnghack(ml[2],ml[1]);
            end if;
            for i from 1 to N do
                sig,a := sigl[i],cl[i];
                if(a>=c1) then
                    next;
                end if;
                l := nops(sig);
                pl := [seq(vmap(x),x=sig)];
                pl := [seq([seq((p[j]-al[j])/(bl[j]-al[j])*ml[j],j=1..d)],p=pl)];
                col := cmap((a-c0)/(c1-c0));
                if(l=1) then
                    SolidCircle(im,op(pl[1]),vertsize,color='black');
                elif(l=2) then
                    Line(im,op(pl[1]),op(pl[2]),color=col,thickness=3*edgesize);
                    Line(im,op(pl[1]),op(pl[2]),color='black',thickness=edgesize);
                elif(l=3) then
                    Poly(im,[pl[1],pl[2],pl[3],pl[1]],pattern="solid",color=col,fill_color=col,fill_pattern="solid");
                    Line(im,op(pl[1]),op(pl[2]),pattern="solid",color=col,thickness=3*edgesize);
                    Line(im,op(pl[1]),op(pl[2]),pattern="solid",color='black',thickness=edgesize);
                    Line(im,op(pl[1]),op(pl[3]),pattern="solid",color=col,thickness=3*edgesize);
                    Line(im,op(pl[1]),op(pl[3]),pattern="solid",color='black',thickness=edgesize);
                    Line(im,op(pl[2]),op(pl[3]),pattern="solid",color=col,thickness=3*edgesize);
                    Line(im,op(pl[2]),op(pl[3]),pattern="solid",color='black',thickness=edgesize);
                    SolidCircle(im,op(pl[1]),vertsize,color='black');
                    SolidCircle(im,op(pl[2]),vertsize,color='black');
                    SolidCircle(im,op(pl[3]),vertsize,color='black');
                end if;
            end do;
            if(channels=4) then
                im := topng(im);
            end if;
            Embed(im);
            return im;
        end proc;
        ModuleApply::static := proc()
            setplex(args);
            return draw(1000);
        end proc;
        `?[]`::static := proc()
            return draw[op(args[2])];
        end proc;
        getcolor::static := proc()
            return cmap;
        end proc;
        setcolor::static := proc(col)
            cmap := colormap(col);
            return;
        end proc;
        reset::static := proc()
            vertsize := 8;
            edgesize := 2;
            channels := 3;
            d := 0;
            setcolor('white');
        end proc;
        reset();
    end module;

    diffshape0 := proc(A::Array(datatype=float[8]),t::float[8],a1::float[8],B::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for i from 1 to m do
            for j from 1 to n do
                a := A[n-j+1,i];
                if(a=Float(infinity)) then
                    next;
                end if;
                r := sqrt(max(a1-a,0.0));
                x := evalf(i)*t;
                y := evalf(j)*t;
                i0 := floor((x-r)/t);
                i1 := ceil((x+r)/t);
                j0 := floor((y-r)/t);
                j1 := ceil((y+r)/t);
                for k from max(i0,1) to min(i1,m) do
                    for l from max(j0,1) to min(j1,n) do
                        x1 := evalf(k)*t;
                        y1 := evalf(l)*t;
                        b := (x1-x)*(x1-x)+(y1-y)*(y1-y)+a;
                        if(b<=a1) then
                            B[n-l+1,k] := min(B[n-l+1,k],b);
                        end if;
                    end do;
                end do;
            end do;
        end do;
    end proc;

    diffshape0 := Compiler:-Compile(diffshape0);

    maxheat := proc(A)
        m,n := Dimension(A);
        ans := -Float(infinity);
        for i from 1 to m do
            for j from 1 to n do
                a := A[i,j];
                if(a>ans and a<>Float(infinity)) then
                    ans := a;
                end if;
            end do;
        end do;
        return ans;
    end proc;

    diffshape := proc(A,t,a1)
        if(nargs=1) then
            return procname(A,1.0);
        elif(nargs=2) then
            return procname(A,t,maxheat(A));
        end if;
        n,m := Dimension(A);
        B := matf(n,m);
        ArrayTools:-Fill(Float(infinity),B);
        diffshape0(A,t,a1,B,m,n);
        return B;
    end proc;

    heatline0 := proc(A::Array(datatype=float[8]),p1::float[8],p2::float[8],q1::float[8],q2::float[8],c0::float[8],c1::float[8],c2::float[8],m::integer[4],n::integer[4])
        N := ceil(max(abs(q1-p1),abs(q2-p2)))+1;
        eps := 1/evalf(N);
        for k from 0 to N do
            x := p1+k*(q1-p1)*eps;
            y := p2+k*(q2-p2)*eps;
            i := round(x);
            j := round(y);
            if(i<1 or i>m or j<1 or j>n) then
                next;
            end if;
            A[n-j+1,i] := c0+c1*x+c2*y;
        end do;
    end proc;

    heatline0 := Compiler:-Compile(heatline0);

    heatline := proc(p,q,a,b,A)
        c0,c1,c2 := simpeqs([p,q],[a,b]);
        n,m := Dimension(A);
        heatline0(A,op(p),op(q),c0,c1,c2,m,n);
        return;
    end proc;

    heatpoly0 := proc(A::Array(datatype=float[8]),p1::float[8],p2::float[8],q1::float[8],q2::float[8],r1::float[8],r2::float[8],z1::float[8],z2::float[8],z3::float[8],m::integer[4],n::integer[4])
        x0 := min(p1,q1,r1);
        x1 := max(p1,q1,r1);
        y0 := min(p2,q2,r2);
        y1 := max(p2,q2,r2);
        i0 := floor(x0);
        i1 := ceil(x1);
        j0 := floor(y0);
        j1 := ceil(y1);
        d := p1*q2-p1*r2-p2*q1+p2*r1+q1*r2-q2*r1;
        eps := 1/evalf(max(i1-i0,j1-j0)+1);
        for i from max(i0,1) to min(i1,m) do
            x := evalf(i);
            for j from max(j0,1) to min(j1,n) do
                y := evalf(j);
                t1 := (q1*r2-q2*r1)+(q2-r2)*x+(r1-q1)*y;
                t2 := (p2*r1-p1*r2)+(r2-p2)*x+(p1-r1)*y;
                t3 := (p1*q2-p2*q1)+(p2-q2)*x+(q1-p1)*y;
                t1 := t1/d;
                t2 := t2/d;
                t3 := t3/d;
                if(t1<-eps or t2<-eps or t3<-eps) then
                    next;
                end if;
                A[n-j+1,i] := min(A[n-j+1,i],t1*z1+t2*z2+t3*z3);
            end do;
        end do;
    end proc;

    heatpoly0 := Compiler:-Compile(heatpoly0);

    heatpoly := proc(p,q,r,a,b,c,A)
        n,m := Dimension(A);
        heatpoly0(A,op(p),op(q),op(r),a,b,c,m,n);
        return;
    end proc;

    heatpoint := proc(p,a,A)
        n,m := Dimension(A);
        i0 := floor(p[1]);
        i1 := ceil(p[1]);
        j0 := floor(p[2]);
        j1 := ceil(p[2]);
        i0 := max(min(i0,m),1);
        i1 := max(min(i1,m),1);
        j0 := max(min(j0,n),1);
        j1 := max(min(j1,n),1);
        A[n-j0+1,i0] := a;
        A[n-j1+1,i0] := a;
        A[n-j0+1,i1] := a;
        A[n-j1+1,i1] := a;
    end proc;

#returns a heat map of the complex with coordinates given by vmap.
    heatshape := proc(X,vmap,N,rng,filt)
        if(type(vmap,'Matrix')) then
            return procname(X,vecmap(X:-getverts(),args[2]),args[3..nargs]);
        end if;
        S := vmap:-getmat();
        if(nargs=2) then
            return procname(X,vmap,1000);
        elif(nargs=3) then
            return procname(X,vmap,N,1.0);
        elif(type(args[4],'numeric')) then
            t := args[4];
            rng1 := drange(S,false);
            rng1 := [seq((op(1,r)-t)..(op(2,r)+t),r=rng1)];
            return procname(X,vmap,N,rng1,args[5..nargs]);
        end if;
        sigl := [seq(op(X[k]),k=0..2)];
        al := [seq(op(1,rng[i]),i=1..2)];
        bl := [seq(op(2,rng[i]),i=1..2)];
        ml := drawdims(N,rng);
        ans := matf(ml[2],ml[1]);
        ArrayTools:-Fill(Float(infinity),ans);
        for sig in sigl do
            l := nops(sig);
            pl := [seq(vmap(x),x=sig)];
            pl := [seq([seq((p[j]-al[j])/(bl[j]-al[j])*ml[j],j=1..2)],p=pl)];
            if(l=1) then
                i1 := op(sig);
                p1 := pl[1];
                a1 := X:-f([i1]);
                heatpoint(p1,a1,ans);
            elif(l=2) then
                i1,i2 := op(sig);
                p1,p2 := pl[1],pl[2];
                a1,a2 := X:-f([i1]),X:-f([i2]);
                p12 := (p1+p2)/2;
                a12 := X:-f([i1,i2]);
                heatline(p1,p12,a1,a12,ans);
                heatline(p2,p12,a2,a12,ans);
            elif(l=3) then
                i1,i2,i3 := op(sig);
                p1,p2,p3 := pl[1],pl[2],pl[3];
                a1,a2,a3 := X:-f([i1]),X:-f([i2]),X:-f([i3]);
                p12,p13,p23 := (p1+p2)/2,(p1+p3)/2,(p2+p3)/2;
                a12,a13,a23 := X:-f([i1,i2]),X:-f([i1,i3]),X:-f([i2,i3]);
                p123 := (p1+p2+p3)/3;
                a123 := X:-f([i1,i2,i3]);
                heatpoly(p1,p12,p123,a1,a12,a123,ans);
                heatpoly(p2,p12,p123,a2,a12,a123,ans);
                heatpoly(p1,p13,p123,a1,a13,a123,ans);
                heatpoly(p3,p13,p123,a3,a13,a123,ans);
                heatpoly(p2,p23,p123,a2,a23,a123,ans);
                heatpoly(p3,p23,p123,a3,a23,a123,ans);
            end if;
        end do;
        if(nargs=5) then
            r := args[5];
            if(r=true) then
                return ans;
            end if;
            if(type(r,'numeric')) then
                a0 := min(map(X:-f,X[0]));
                a1 := r;
            else
                a0,a1 := op(r);
            end if;
        else
            V := map(X:-f,X[0]);
            a0,a1 := min(V),max(V);
        end if;
        return heatmap(ans,a0..a1);
    end proc;

    powheat0 := proc(A::Array(datatype=float[8]),pow::Array(datatype=float[8]),a1::float[8],x0::float[8],y0::float[8],dx::float[8],dy::float[8],B::Array(datatype=float[8]),N::integer[4],m::integer[4],n::integer[4])
        for k from 1 to N do
            x := A[k,1];
            y := A[k,2];
            r := sqrt(max(a1+pow[k],0.0));
            i0 := floor(.5+(x-r-x0)/dx);
            i1 := ceil(.5+(x+r-x0)/dx);
            j0 := floor(.5+(y-r-y0)/dy);
            j1 := ceil(.5+(y+r-y0)/dy);
            for i from max(i0,1) to min(i1,m) do
                for j from max(j0,1) to min(j1,n) do
                    x1 := x0+(i-.5)*dx;
                    y1 := y0+(j-.5)*dy;
                    b := (x1-x)*(x1-x)+(y1-y)*(y1-y)-pow[k];
                    if(b<=a1) then
                        B[n-j+1,i] := min(B[n-j+1,i],b);
                    end if;
                end do;
            end do;
        end do;
    end proc;

    powheat0 := Compiler:-Compile(powheat0);

#draw the heat map of a power diagram
    powheat := proc(A,pow,a1,rng)
    local c;
        if(type(a1,'numeric')) then
            c0,c1 := min(-pow),a1;
        else
            c0,c1 := op(a1);
        end if;
        if(nargs=3) then
            return procname(args,powrange(A,pow,c1));
        end if;
        if(not type(procname,indexed) or nops(procname)=0) then
            return powheat[1000](args);
        end if;
        N := Dimension(A)[1];
        al,bl := endrange(rng);
        x0,y0 := op(al);
        ml,tl := rastrange(rng,op(procname));
        m,n := op(ml);
        dx,dy := op(tl);
        B := allocla[float[8]]([n,m]);
        ArrayTools:-Fill(Float(infinity),B);
        powheat0(A,pow,c1,x0,y0,dx,dy,B,N,m,n);
        map[inplace](c->(c-c0)/(c1-c0),B);
        return heatmat(B);
    end proc;

    powrange := proc(A,pow,a1)
        N,m := Dimension(A);
        kl := [];
        for k from 1 to N do
            if(-pow[k]<a1) then
                kl := [op(kl),k];
            end if;
        end do;
        ans := [];
        for j from 1 to m do
            x0 := min(seq(A[k,j]-sqrt(max(a1+pow[k],0.0)),k=kl));
            x1 := max(seq(A[k,j]+sqrt(max(a1+pow[k],0.0)),k=kl));
            ans := [op(ans),x0..x1];
        end do;
        return ans;
    end proc;

    endrange := proc(rng)
        d := nops(rng);
        al := [seq(evalf(op(1,rng[i])),i=1..d)];
        bl := [seq(evalf(op(2,rng[i])),i=1..d)];
        return al,bl;
    end proc;

    #flag=true/false means use max/min as the reference size
    rastrange := proc(rng,M,flag:=true)
        d := nops(rng);
        al,bl := endrange(rng);
        cl := bl-al;
        if(type(M,'numeric')) then
            sig := sort(cl,output=permutation);
            if(flag) then
                i := sig[d];
            else
                i := sig[1];
            end if;
            ml := [seq(flag,i=1..d)];
            ml[i] := M;
            return procname(rng,ml);
        end if;
        ml := M;
        tl := [];
        for i from 1 to d do
            if(type(ml[i],'numeric')) then
                tl := [op(tl),evalf(cl[i]/ml[i])];
            end if;
        end do;
        t0,t1 := min(tl),max(tl);
        dxl := [seq(0.0,i=1..d)];
        for i from 1 to d do
            if(ml[i]=true) then
                ml[i] := ceil(cl[i]/t1);
            elif(ml[i]=false) then
                ml[i] := ceil(cl[i]/t0);
            end if;
            dxl[i] := evalf(cl[i]/ml[i]);
        end do;
        return ml,dxl;
    end proc;

    drawalpha0 := proc(im1::Array(datatype=float[8]),im2::Array(datatype=float[8]),m::integer[4],n::integer[4])
        for i from 1 to m do
            for j from 1 to n do
                if(im2[i,j,1]<=1.0) then
                    im1[i,j,4] := 0.0;
                    for k from 1 to 3 do
                        im1[i,j,k] := im2[i,j,k];
                    end do;
                end if;
            end do
        end do;
    end proc;

    drawalpha0 := Compiler:-Compile(drawalpha0);

#draw the power diagram with the alpha complex overlayed. uses the
#coloring system defined in the heatmat/drawplex objects. d can be the
#desired degree, or a complex itself. in the case of a complex, the
#power diagram will use only those indices which correspond to actual
#0-simplices, so that the two parts of the diagram are always homotopy
#equivalent.
    drawalpha := proc(S,pow,a1,d,rng)
        if(nargs=4) then
            return procname(S,pow,a1,d,powrange(S,pow,a1));
        elif(not type(procname,indexed)) then
            return drawalpha[2000](args);
        end if;
        ml,tl := rastrange(rng,op(procname));
        if(type(a1,'numeric')) then
            c1 := a1;
            c0 := -max(pow);
        else
            c0,c1 := op(a1);
        end if;
        if(type(args[4],'numeric')) then
            X := alphaplex[true](S,pow,c1,d)[1];
        else
            X := args[4];
        end if;
        il := map(op,X[0]);
        im1 := powheat[ml](S[il],pow[il],c0..c1,rng);
        drawplex:-channels := 3;
        drawplex:-setplex(X,S,c0..c1,rng);
        im2 := drawplex:-draw(ml);
        drawalpha0(im1,im2,ml[2],ml[1]);
        Embed(im1);
        return im1;
    end proc;

end module;
