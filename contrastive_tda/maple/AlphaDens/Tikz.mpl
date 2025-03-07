Tikz := module()
option package;
export tikzdata,tikzdens,tikzcloud,tikzpd,tikzbary,tikzplex,getproj,savetikz,tikzmulti,tikzcolor,tikzrange,tikzaxes,tikzscatter;

    tikzdata := proc(Al,colors)
        if(type(procname,indexed)) then
            md := tikzdens(args);
            return md(op(procname));
        elif(not type(args[1],'list')) then
            return tikzdata([args[1]],seq([col],col=args[2..nargs]));
        end if;
        l := nops(Al);
        if(nargs=1) then
            if(l>5) then
                error;
            elif(l=1) then
                return tikzdata(Al,["black"]);
            else
                return tikzdata(Al,["blue","red","green","orange","brown"]);
            end if;
        end if;
        argl := [Al,colors];
        md := module()
        option object;
        export Al,colors,l,out,tofile,nodethickness;
        local ModulePrint,ModuleApply;
            ModulePrint::static := proc()
                return nprintf("tikz data set");
            end proc;
            ModuleApply::static := proc()
                if(nargs=0) then
                    return out();
                else
                    return tofile(args);
                end if;
            end proc;
            out::static := proc()
                ans := [];
                for k from 1 to l do
                    A := Al[k];
                    col := colors[k];
                    N := Dimension(A)[1];
                    for i from 1 to N do
                        ans := cat(ans,nprintf(cat("\\filldraw[color=",col,"] (%f,%f) circle (%dpt);\n"),A[i,1],A[i,2],round(nodethickness)));
                    end do;
                end do;
                return ans;
            end proc;
            tofile::static := proc(fn)
                savetikz[op(getsubs(procname))](fn,out());
            end proc;
            Al,colors := op(argl);
            l := nops(Al);
            nodethickness := 1;
        end module;
    end proc;

#make a tikz file with the coordinates of a filtered complex
    tikzplex := proc(X,vmap,fn)
        if(nargs>2) then
            md := tikzplex(X,vmap);
            return md:-tofile(fn);
        elif(type(vmap,'Matrix')) then
            return procname(X,vecmap(X:-getverts(),vmap));
        end if;
        argl := [X,vmap,color];
        md := module()
        option object;
        export X,vmap,nodethick,edgethick,color,c0,c1,init,tofile,sigl,getvert;
        local ModulePrint,ModuleApply;
            ModulePrint::static := proc()
                return nprintf("tikz plex");
            end proc;
            out::static := proc()
                ans := "";
                for sig in sigl do
                    c := X:-f(sig);
                    k := max(0,round(1000*(c-c0)/(c1-c0)));
                    if(k>1000) then
                        next;
                    end if;
                    k := 1000-k;
                    col := nprintf(cat("color of colormap={%d of ",color,"}"),k);
                    if(nops(sig)=1) then
                        p1 := getvert(sig[1]);
                        ans := cat(ans,nprintf(cat("\\fill[",col,"] (%f,%f) circle[shift only,radius=(%dpt)];\n"),op(p1),nodethick));
                    elif(nops(sig)=2) then
                        p1,p2 := seq(getvert(i),i=sig);
                        ans := cat(ans,nprintf(cat("\\draw[",col,",",edgethick,"] (%f,%f)--(%f,%f);\n"),op(p1),op(p2)));
                    elif(nops(sig)=3) then
                        p1,p2,p3 := seq(getvert(i),i=sig);
                        ans := cat(ans,nprintf(cat("\\fill[",col,"] (%f,%f)--(%f,%f)--(%f,%f);\n"),op(p1),op(p2),op(p3)));
                    end if;
                end do;
                return nprintf(ans);
            end proc;
            getvert::static := proc(x)
                return vmap(x)[1..2];
            end proc;
            tofile::static := proc(fn)
                savetikz(fn,out());
            end proc;
            init::static := proc()
                X,vmap := args;
                sigl := [seq(op(X[k]),k=0..2)];
                ord := sort(map(X:-f,sigl),`>`,output=permutation);
                sigl := sigl[ord];
                N := nops(sigl);
                cl := map(X:-f,X[0]);
                c0,c1 := min(cl),max(cl);
                nodethick := 1;
                edgethick := "semithick";
                color := "colormap/viridis";
            end proc;
        end module;
        md:-init(X,vmap);
        return md;
    end proc;

    tikzcloud := proc(A)
        if(not type(A,'Matrix')) then
            return procname(Matrix(A,datatype=float[8]),args[2..nargs]);
        end if;
        md := module()
        option object;
        export A,N,m,out,tofile,vertsize,coords,init,color;
            out::static := proc()
                ans := "";
                for i from 1 to N do
                    p := [A[i,coords[1]],A[i,coords[2]]];
                    ans := cat(ans,nprintf(cat("\\fill[color=",color,"] (%f,%f) circle[shift only,radius=(%dpt)];\n"),op(p),vertsize));
                end do;
                return ans;
            end proc;
            tofile::static := proc(fn)
                savetikz[op(getsubs(procname))](fn,out(args[2..nargs]));
            end proc;
            init::static := proc()
                A := args;
                N,m := Dimension(A);
                vertsize := 1;
                coords := [1,2];
                color := "black";
            end proc;
        end module;
        md:-init(A);
        return md;
    end proc;

    tikzscatter := proc(A)
        if(not type(A,'Matrix')) then
            return procname(Matrix(A,datatype=float[8]),args[2..nargs]);
        end if;
        md := module()
        option object;
        export A,N,m,out,tofile,vertsize,coords,init,color;
            out::static := proc()
                ans := "\\begin{axis}[scatter/classes={\n a={mark=*,draw=black,mark size=.5}}]\n \\addplot[scatter,only marks,\n scatter src=explicit symbolic]\n table[meta=label] {\n,x y label\n";
                for i from 1 to N do
                    p := [A[i,coords[1]],A[i,coords[2]]];
                    ans := cat(ans,nprintf(cat("%f %f a\n"),op(p)));
                end do;
                ans := cat(ans,"};\n\\end{axis}\n");
                return ans;
            end proc;
            tofile::static := proc(fn)
                savetikz[op(getsubs(procname))](fn,out(args[2..nargs]));
            end proc;
            init::static := proc()
                A := args;
                N,m := Dimension(A);
                vertsize := 1;
                coords := [1,2];
                color := "black";
            end proc;
        end module;
        md:-init(A);
        return md;
    end proc;

    tikzpd := proc(pd,c1:=pd:-a1)
        if(type(procname,indexed)) then
            md := tikzpd(args);
            return md(op(procname));
        end if;
        argl := [pd,c1];
        md := module()
        option object;
        export pd,L,n,X,W,c0,c1,tofile,colormap,getcolor,out,out0,out1,out2,out3,rev,thickness,nodethickness,bordercolor,getverts;
        local ModulePrint,ModuleApply;
            ModulePrint::static := proc()
                return nprintf("tikz power diagram");
            end proc;
            ModuleApply::static := proc()
                if(nargs=0) then
                    return out();
                else
                    return tofile(args);
                end if;
            end proc;
            out::static := proc(typ:='ALL')
                s0 := out0(args);
                s1 := out1(args);
                s2 := out2(args);
                s3 := out3(args);
                if(typ='ALL') then
                    return nprintf(cat(s1,s2,s3,s0));
                elif(typ='UNION') then
                    return nprintf(cat(s1,s2));
                end if;
            end proc;
            out0::static := proc()
                ans := "";
                for k from 1 to n do
                    z := [L[k,1],L[k,2]];
                    col := bordercolor;
                    ans := cat(ans,nprintf(cat("\\filldraw[",col,"] (%f,%f) circle (",nodethickness,");\n"),op(z)));
                end do;
                return ans;
            end proc;
            out1::static := proc()
                ans := "";
                for k in X:-verts() do
                    r := sqrt(c1+pd:-pow[k]);
                    z := [L[k,1],L[k,2]];
                    sigl := star([k],X,1);
                    col := getcolor(fof([k]));
                    if(nops(sigl)=0) then
                        ans1 := nprintf(cat("\\filldraw[",col,"] (%f,%f) circle (%f);"),op(z),r);
                    else
                        ql,arcs,el,vals := getverts(k);
                        l := nops(ql)-1;
                        ans1 := nprintf(cat("\\filldraw[",col,"] (%f,%f)"),op(ql[1]));
                        for i from 1 to l do
                            if(arcs[i]) then
                                ans1 := cat(ans1,tkarc(z,r,ql[i],ql[i+1]));
                            else
                                ans1 := cat(ans1,nprintf("--(%f,%f)",op(ql[i+1])));
                            end if;
                        end do;
                    end if;
                    ans := cat(ans,ans1,";\n");
                end do;
                return ans;
            end proc;
            out2::static := proc()
                ans := "";
                for k in X:-verts() do
                    r := sqrt(c1+pd:-pow[k]);
                    z := [L[k,1],L[k,2]];
                    sigl := star([k],X,1);
                    if(nops(sigl)=0) then
                        col := bordercolor;
                        ans1 := nprintf(cat("\\draw[",col,",",thickness,"] (%f,%f) circle (%f);"),op(z),r);
                        ans := cat(ans,ans1);
                    else
                        ql,arcs,el,vals := getverts(k);
                        l := nops(ql)-1;
                        for i from 1 to l do
                            if(arcs[i]) then
                                col := bordercolor;
                                ans1 := nprintf(cat("\\draw[",col,",",thickness,"] (%f,%f)"),op(ql[i]));
                                ans1 := cat(ans1,tkarc(z,r,ql[i],ql[i+1]));
                                ans := cat(ans,ans1,";\n");
                            end if;
                        end do;
                    end if;
                end do;
                return ans;
            end proc;
            out3::static := proc()
                ans := "";
                for k in X:-verts() do
                    r := sqrt(c1+pd:-pow[k]);
                    z := [L[k,1],L[k,2]];
                    sigl := star([k],X,1);
                    if(nops(sigl)=0) then
                        next;
                    else
                        ql,arcs,el,vals := getverts(k);
                        l := nops(ql)-1;
                        for i from 1 to l do
                            if(not arcs[i]) then
                                col := bordercolor;
                                ans1 := nprintf(cat("\\draw[",col,",",thickness,"] (%f,%f)"),op(ql[i]));
                                ans1 := cat(ans1,nprintf("--(%f,%f)",op(ql[i+1])));
                                ans := cat(ans,ans1,";\n");
                            end if;
                        end do;
                    end if;
                end do;
                return ans;
            end proc;
            getcolor::static := proc(c)
                t := (c-c0)/(c1-c0);
                k := min(max(round(1000*t),0),1000);
                if(not rev) then
                    k := 1000-k;
                end if;
                col := nprintf(cat("color of colormap={%d of colormap/",colormap,"}"),k);
                return col;
            end proc;
            getverts::static := proc(k)
                print(hi2);
                r := sqrt(pd:-maxpow+pd:-pow[k]);
                z := [L[k,1],L[k,2]];
                sigl := star([k],X,1);
                il := [seq(({op(sig)} minus {k})[1],sig=sigl)];
                l := nops(sigl);
                zl := [seq([L[il[j],1],L[il[j],2]],j=1..l)];
                ul := [seq(zl[i]-z,i=1..l)];
                wl := [seq([-u[2],u[1]]/lnorm(u),u=ul)];
                pl := [seq(W(sort([k,i])),i=il)];
                vl := [seq(pl[i]-z,i=1..l)];
                ord := ordangles(wl);
                ord := [ord[l],op(ord)];
                ql,arcs,el,vals := [],[],[],[];
                print(hi1);
                for j from 0 to l do
                    i1 := ord[j+1];
                    i2 := ord[j+1 mod l+1];
                    z1,z2 := zl[i1],zl[i2];
                    u1,u2 := ul[i1],ul[i2];
                    p1,p2 := pl[i1],pl[i2];
                    w1,w2 := wl[i1],wl[i2];
                    v1,v2 := vl[i1],vl[i2];
                    sig := [op({op([k,il[i1],il[i2]])})];
                    if(nops(sig)=3 and X:-X:-contains(sig) and ldot(w1,z1-z2)<0) then
                        isarc := false;
                        q := W(sig);
                    else
                        isarc := true;
                        t1 := -ldot(v1,w1)+sqrt(ldot(v1,w1)^2+r^2-lnorm(v1)^2);
                        q1 := p1+t1*w1;
                        t2 := -ldot(v2,w2)-sqrt(ldot(v2,w2)^2+r^2-lnorm(v2)^2);
                        q := p2+t2*w2;
                    end if;
                    if(j=0) then
                        ql := [op(ql),q];
                    elif(isarc) then
                        ql := [op(ql),q1,q];
                        arcs := [op(arcs),false,true];
                        el := [op(el),[p1,w1]];
                        vals := [op(vals),X(sig),c1];
                    elif(not isarc) then
                        ql := [op(ql),q];
                        arcs := [op(arcs),false];
                        el := [op(el),[p1,w1]];
                        vals := [op(vals),X(sig)];
                    end if;
                end do;
                print(hi3);
                return ql,arcs,el,vals;
            end proc;
            tofile::static := proc(fn)
                savetikz[op(getsubs(procname))](fn,out(args[2..nargs]));
            end proc;
            pd,c1 := op(argl);
            L := pd:-A;
            n := pd:-n;
            #X,W := alphaplex['ALPHA','WITS'](pd,c1,2);
            X,W := getalpha(pd,2);
            c0 := min(seq(X(sig),sig=X[0]));
            #c1 := pd:-maxpow;
            colormap := "viridis";
            thickness := "thick";
            nodethickness := "1pt";
            bordercolor := "black";
            rev := false;
        end module;
        return md;
    end proc;

    star := proc(sig,X,d)
        sigl := X[d];
        ans := [];
        for sig1 in sigl do
            if({op(sig)} subset {op(sig1)}) then
                ans := [op(ans),sig1];
            end if;
        end do;
        return ans;
    end proc;

    tikzpd := proc(pd,c1:=pd:-a1)
        if(type(procname,indexed)) then
            md := tikzpd(args);
            return md(op(procname));
        end if;
        argl := [pd,c1];
        md := module()
        option object;
        export pd,L,n,X,W,c0,c1,tofile,colormap,getcolor,out,out0,out1,out2,out3,rev,thickness,nodethickness,bordercolor,getverts,contains,contains0;
        local ModulePrint,ModuleApply;
            ModulePrint::static := proc()
                return nprintf("tikz power diagram");
            end proc;
            ModuleApply::static := proc()
                if(nargs=0) then
                    return out();
                else
                    return tofile(args);
                end if;
            end proc;
            out::static := proc(typ:='ALL')
                s0 := out0(args);
                s1 := out1(args);
                s2 := out2(args);
                s3 := out3(args);
                if(typ='ALL') then
                    return nprintf(cat(s1,s2,s3,s0));
                elif(typ='UNION') then
                    return nprintf(cat(s1,s2));
                end if;
            end proc;
            out0::static := proc()
                ans := "";
                for k from 1 to n do
                    z := [L[k,1],L[k,2]];
                    col := bordercolor;
                    ans := cat(ans,nprintf(cat("\\filldraw[",col,"] (%f,%f) circle (",nodethickness,");\n"),op(z)));
                end do;
                return ans;
            end proc;
            out1::static := proc()
                ans := "";
                for k in X:-getverts() do
                    if(not contains([k])) then
                        next;
                    end if;
                    r := sqrt(c1+pd:-pow[k]);
                    z := [L[k,1],L[k,2]];
                    sigl := star([k],X,1);
                    col := getcolor(fof([k]));
                    if(nops(sigl)=0) then
                        ans1 := nprintf(cat("\\filldraw[",col,"] (%f,%f) circle (%f);"),op(z),r);
                    else
                        ql,arcs,el,vals := getverts(k);
                        l := nops(ql)-1;
                        ans1 := nprintf(cat("\\filldraw[",col,"] (%f,%f)"),op(ql[1]));
                        for i from 1 to l do
                            if(arcs[i]) then
                                ans1 := cat(ans1,tkarc(z,r,ql[i],ql[i+1]));
                            else
                                ans1 := cat(ans1,nprintf("--(%f,%f)",op(ql[i+1])));
                            end if;
                        end do;
                    end if;
                    ans := cat(ans,ans1,";\n");
                end do;
                return ans;
            end proc;
            out2::static := proc()
                ans := "";
                for k in X:-getverts() do
                    if(not contains([k])) then
                        next;
                    end if;
                    r := sqrt(c1+pd:-pow[k]);
                    z := [L[k,1],L[k,2]];
                    sigl := star([k],X,1);
                    if(nops(sigl)=0) then
                        col := bordercolor;
                        ans1 := nprintf(cat("\\draw[",col,",",thickness,"] (%f,%f) circle (%f);"),op(z),r);
                        ans := cat(ans,ans1);
                    else
                        ql,arcs,el,vals := getverts(k);
                        l := nops(ql)-1;
                        for i from 1 to l do
                            if(arcs[i]) then
                                col := bordercolor;
                                ans1 := nprintf(cat("\\draw[",col,",",thickness,"] (%f,%f)"),op(ql[i]));
                                ans1 := cat(ans1,tkarc(z,r,ql[i],ql[i+1]));
                                ans := cat(ans,ans1,";\n");
                            end if;
                        end do;
                    end if;
                end do;
                return ans;
            end proc;
            out3::static := proc()
                ans := "";
                for k in X:-getverts() do
                    r := sqrt(c1+pd:-pow[k]);
                    z := [L[k,1],L[k,2]];
                    sigl := star([k],X,1);
                    if(nops(sigl)=0) then
                        next;
                    else
                        ql,arcs,el,vals := getverts(k);
                        l := nops(ql)-1;
                        for i from 1 to l do
                            if(not arcs[i]) then
                                col := bordercolor;
                                ans1 := nprintf(cat("\\draw[",col,",",thickness,"] (%f,%f)"),op(ql[i]));
                                ans1 := cat(ans1,nprintf("--(%f,%f)",op(ql[i+1])));
                                ans := cat(ans,ans1,";\n");
                            end if;
                        end do;
                    end if;
                end do;
                return ans;
            end proc;
            getcolor::static := proc(c)
                t := (c-c0)/(c1-c0);
                k := min(max(round(1000*t),0),1000);
                if(not rev) then
                    k := 1000-k;
                end if;
                col := nprintf(cat("color of colormap={%d of colormap/",colormap,"}"),k);
                #col := nprintf("pink");
                return col;
            end proc;
            getverts::static := proc(k)
                r := sqrt(pd:-a1+pd:-pow[k]);
                z := [L[k,1],L[k,2]];
                sigl := star([k],X,1);
                il := [seq(({op(sig)} minus {k})[1],sig=sigl)];
                l := nops(sigl);
                zl := [seq([L[il[j],1],L[il[j],2]],j=1..l)];
                ul := [seq(zl[i]-z,i=1..l)];
                wl := [seq([-u[2],u[1]]/lnorm(u),u=ul)];
                pl := [seq(convert(W(sort([k,i])),'list'),i=il)];
                vl := [seq(pl[i]-z,i=1..l)];
                ord := ordangles(wl);
                ord := [ord[l],op(ord)];
                ql,arcs,el,vals := [],[],[],[];
                for j from 0 to l do
                    i1 := ord[j+1];
                    i2 := ord[j+1 mod l+1];
                    z1,z2 := zl[i1],zl[i2];
                    u1,u2 := ul[i1],ul[i2];
                    p1,p2 := pl[i1],pl[i2];
                    w1,w2 := wl[i1],wl[i2];
                    v1,v2 := vl[i1],vl[i2];
                    sig := [op({op([k,il[i1],il[i2]])})];
                    if(nops(sig)=3 and contains(sig) and ldot(w1,z1-z2)<0) then
                        isarc := false;
                        q := convert(W(sig),'list');
                    else
                        isarc := true;
                        t1 := -ldot(v1,w1)+sqrt(ldot(v1,w1)^2+r^2-lnorm(v1)^2);
                        q1 := p1+t1*w1;
                        t2 := -ldot(v2,w2)-sqrt(ldot(v2,w2)^2+r^2-lnorm(v2)^2);
                        q := p2+t2*w2;
                    end if;
                    if(j=0) then
                        ql := [op(ql),q];
                    elif(isarc) then
                        ql := [op(ql),q1,q];
                        arcs := [op(arcs),false,true];
                        el := [op(el),[p1,w1]];
                        vals := [op(vals),fof(sig),c1];
                    elif(not isarc) then
                        ql := [op(ql),q];
                        arcs := [op(arcs),false];
                        el := [op(el),[p1,w1]];
                        vals := [op(vals),fof(sig)];
                    end if;
                end do;
                return ql,arcs,el,vals;
            end proc;
            fof::static := proc(sig)
                if(contains(sig)) then
                    return X:-f(sig);
                else
                    return Float(infinity);
                end if;
            end proc;
            contains::static := proc(sig)
                sigl0 := contains0();
                if(sig in sigl0) then
                    return true;
                else
                    return false;
                end if;
            end proc;
            contains0::static := proc()
            option remember;
                return [seq(op(X[k]),k=0..2)];
            end proc;
            tofile::static := proc(fn)
                savetikz[op(getsubs(procname))](fn,out(args[2..nargs]));
            end proc;
            pd,c1 := op(argl);
            L := pd:-A;
            n := pd:-N;
            #X,W := alphaplex['ALPHA','WITS'](pd,c1,2);
            X,W := getalpha(pd:-A,pd:-pow,c1,2);
            c0 := min(seq(fof(sig),sig=X[0]));
            colormap := "viridis";
            thickness := "thick";
            nodethickness := "1pt";
            bordercolor := "black";
            rev := false;
        end module;
        return md;
    end proc;

    ldot := proc(u,v)
        n := nops(u);
        return add(u[i]*v[i],i=1..n);
    end proc;

    lnorm := proc(u)
        return sqrt(ldot(u,u));
    end proc;

    orient := proc(p1,p2,p3)
        u := p2-p1;
        v := p3-p1;
        return u[1]*v[2]-u[2]*v[1];
    end proc;

    tkarc := proc(z,r,p,q)
        tt0 := 180.0/Pi*arccos((p[1]-z[1])/r);
        if(p[2]-z[2]<0) then
            tt0 := -tt0;
        end if;
        c := sqrt(ldot(p-q,p-q));
        tt1 := 180.0/Pi*arccos(1-c^2/2/r^2);
        if(orient(z,p,q)<0) then
            tt1 := 360-tt1;
        end if;
        s := " arc (%f:%f:%f)";
        return nprintf(s,tt0,tt0+tt1,r);
    end proc;

    ordangles := proc(pl)
        n := nops(pl);
        if(n<=2) then
            return [seq(i,i=1..n)];
        end if;
        z := add(p,p=pl)/n;
        ans := [];
        for p in pl do
            v := p-z;
            r := sqrt(ldot(v,v));
            tt := evalf(arccos(v[1]/r));
            if(v[2]<0) then
                tt := evalf(2*Pi-tt);
            end if;
            ans := [op(ans),tt];
        end do;
        sig := sort(ans,output=permutation);
        return sig;
    end proc;

    tikzbary := proc(X,W)
        argl := [args];
        md := module()
        option object;
        export X,W,n,c0,c1,nodethickness,edgethickness,color,out,tofile;
            ModulePrint::static := proc()
                return nprintf("tikz alpha plex");
            end proc;
            ModuleApply::static := proc()
                if(nargs=0) then
                    return out();
                else
                    return tofile(args);
                end if;
            end proc;
            out::static := proc()
                ans := "";
                for sig in X[2] do
                    p1,p2,p3,p12,p13,p23,p123 := convert(W(sig[[1]]),'list'),convert(W(sig[[2]]),'list'),convert(W(sig[[3]]),'list'),convert(W(sig[[1,2]]),'list'),convert(W(sig[[1,3]]),'list'),convert(W(sig[[2,3]]),'list'),convert(W(sig[[1,2,3]]),'list');
                    ans := cat(ans,nprintf(cat("\\filldraw[",fillcolor,"] "
                                               "(%f,%f)--(%f,%f)--(%f,%f)--(%f,%f)--(%f,%f)--(%f,%f);\n"),op(p1),op(p12),op(p2),op(p23),op(p3),op(p13)));
                    for p in [p1,p2,p3,p12,p13,p23] do
                        ans := cat(ans,nprintf(cat("\\draw[",color,",",edgethickness,",",edgestyle,"] (%f,%f)--(%f,%f);\n"),op(p123),op(p)));
                    end do;
                end do;
                for sig in X[0] do
                    p1 := convert(W(sig),'list');
                    ans := cat(ans,nprintf(cat("\\filldraw[",color,"] (%f,%f) circle (%fpt);\n"),op(p1),nodethickness));
                end do;
                for sig in X[1] do
                    p1,p2,p12 := convert(W(sig[[1]]),'list'),convert(W(sig[[2]]),'list'),convert(W(sig),'list');
                    for p in [p1,p2] do
                        ans := cat(ans,nprintf(cat("\\draw[",color,",",edgethickness,"] (%f,%f)--(%f,%f);\n"),op(p),op(p12)));
                    end do;
                end do;
                return ans;
            end proc;
            tofile::static := proc(fn)
                savetikz[op(getsubs(procname))](fn,out());
            end proc;
            X,W := op(argl);
            n := nops(X:-getverts());
            nodethickness := 1.0;
            edgethickness := "thin";
            edgestyle := "densely dashed";
            color := "black";
            fillcolor := "pink";
        end module;
    end proc;

    tikzplex1 := proc(X,verts,color:="colormap/viridis")
        if(type(procname,indexed)) then
            md := tikzplex(args);
            return md(op(procname));
        end if;
        argl := [X,verts,color];
        md := module()
        option object;
        export verts,X,color,n,d,nodethickness,edgethickness,out,c0,c1,sigl,tofile;
        local ModulePrint,ModuleApply;
            ModulePrint::static := proc()
                return nprintf("tikz plex");
            end proc;
            ModuleApply::static := proc()
                if(nargs=0) then
                    return out();
                else
                    return tofile(args);
                end if;
            end proc;
            out::static := proc()
                ans := "";
                for sig in sigl do
                    c := X:-f(sig);
                    if(c>c1) then
                        next;
                    end if;
                    k := max(0,round(1000*(c-c0)/(c1-c0)));
                    k := 1000-k;
                    col := nprintf(cat("color of colormap={%d of ",color,"}"),k);
                    if(nops(sig)=1) then
                        p1 := verts(sig[1]);
                        ans := cat(ans,nprintf(cat("\\filldraw[",col,"] (%f,%f) circle (%dpt);\n"),op(p1),nodethickness));
                    elif(nops(sig)=2) then
                        p1,p2 := verts(sig[1]),verts(sig[2]);
                        ans := cat(ans,nprintf(cat("\\draw[",col,",",edgethickness,"] (%f,%f)--(%f,%f);\n"),op(p1),op(p2)));
                    elif(nops(sig)=3) then
                        p1,p2,p3 := verts(sig[1]),verts(sig[2]),verts(sig[3]);
                        ans := cat(ans,nprintf(cat("\\filldraw[",col,"] (%f,%f)--(%f,%f)--(%f,%f);\n"),op(p1),op(p2),op(p3)));
                    end if;
                end do;
                return nprintf(ans);
            end proc;
            tofile::static := proc(fn)
                savetikz(fn,out());
            end proc;
            X,verts,color := op(argl);
            sigl := X[];
            N := nops(sigl);
            sigl := [seq(sigl[N-i+1],i=1..N)];
            c0 := min(map(X,sigl));
            c1 := max(map(X,sigl));
            labs := verts:-labs;
            nodethickness := 2;
            edgethickness := "very thick";
        end module;
        return md;
    end proc;


    tikzmulti := proc(W,fl,colors)
        argl := [args];
        md := module()
        option object;
        export W,fl,colors,l,n,d,nodethickness,edgethickness,color,tikz,getdens,getcolor,d0,sigl,tofile;
        local V1,V2,V3;
            ModulePrint::static := proc()
                return nprintf("tikz multi");
            end proc;
            tikz::static := proc()
                ans := [];
                for sig in sigl do
                    dl := getdens(sig);
                    if(convert(dl,`+`)<d0) then
                        next;
                    end if;
                    col := tikzcolor(dl,colors);
                    if(nops(sig)=1) then
                        W:-getelt[V1]([sig[1]]);
                        ans := [op(ans),nprintf(cat("\\filldraw[color=",col,"] (%f,%f) circle (%fpt);\n"),V1[1],V1[2],nodethickness)];
                    elif(nops(sig)=2) then
                        W:-getelt[V1,V2]([sig[1]],[sig[2]]);
                        ans := [op(ans),nprintf(cat("\\draw[color=",col,",",edgethickness,"] (%f,%f)--(%f,%f);\n"),V1[1],V1[2],V2[1],V2[2])];
                    elif(nops(sig)=3) then
                        W:-getelt[V1,V2,V3]([sig[1]],[sig[2]],[sig[3]]);
                        ans := [op(ans),nprintf(cat("\\filldraw[color=",col,"] (%f,%f)--(%f,%f)--(%f,%f);\n"),V1[1],V1[2],V2[1],V2[2],V3[1],V3[2])];
                    end if;
                end do;
                return ans;
            end proc;
            tofile::static := proc(fn)
                savetikz(fn,tikz());
            end proc;
            getcolor::static := proc(sig)
                dl := getdens(sig);
                return tikzcolor(dl,colors);
            end proc;
            getdens::static := proc(sig)
                W:-getelt[V1](sig);
                ans := [];
                for f in fl do
                    ans := [op(ans),f(V1)];
                end do;
                return ans;
            end proc;
            W,fl,colors := op(argl);
            d0 := 0.0;
            l := nops(Xl);
            sigl := W:-labs;
            N := nops(sigl);
            sigl := [seq(sigl[N-i+1],i=1..N)];
            nodethickness := 1.0;
            edgethickness := "thick";
            V1,V2,V3 := allocla(2,2,2);
        end module;
        return md;
    end proc;

    tikzcolor := proc(dl,colors,bg:="white")
        l := nops(dl);
        pl := [];
        for i from 2 to l do
            d1 := add(dl[j],j=1..i-1);
            d2 := add(dl[j],j=1..i);
            p := round(100*d1/d2);
            if(p<0 or p>100) then
                error "bad color range";
            end if;
            pl := [op(pl),p];
        end do;
        d1 := add(dl[j],j=1..l);
        d2 := 1.0;
        p := round(100*d1/d2);
        if(p>100) then
            error; "bad color range";
        end if;
        ans := convert(colors[1],'string');
        for i from 2 to l do
            ans := cat(ans,"!%d!",colors[i]);
        end do;
        ans := cat(ans,"!%d!",bg);
        return nprintf(ans,op(pl),p);
    end proc;

    discolor := proc(dl,ctab,bg)
        l := nops(dl);
        for i from 1 to l do
            k := round(1000*dl[i]);
            if(k<0 or k>1000) then
                error "out of discolor range";
            end if;
            colors := [op(colors),ctab[i][k]];
        end do;
        return tikzcolor(dl,colors,args[3..nargs]);
    end proc;

    savetikz := proc(fn,s)
        if(not type(procname,indexed) or nops(procname)=0) then
            return savetikz[true](args);
        end if;
        fn1 := cat(fn,".tex");
        e := op(procname);
        fh := fopen(fn1,'WRITE');
        if(e) then
            fprintf(fh,"\\documentclass{standalone}\n");
            fprintf(fh,"\usepackage{xcolor}\n");
            fprintf(fh,"\\usepackage{tikz}\n");
            fprintf(fh,"\\usepackage{pgfplots}\n");
            fprintf(fh,"\\usepackage{pgfplots}\n");
            fprintf(fh,"\\usepgfplotslibrary{colormaps}\n");
            fprintf(fh,"\\pgfplotsset{compat=1.16}\n");
            fprintf(fh,"\\usetikzlibrary{positioning, backgrounds}\n\n");
            fprintf(fh,"\\begin{document}\n\n");
            fprintf(fh,cat("\\begin{tikzpicture}[",args[3..nargs],"]\n\n"));
        end if;
        fprintf(fh,s);
        if(e) then
            fprintf(fh,"\\end{tikzpicture}\n\n");
            fprintf(fh,"\\end{document}\n");
        end if;
        fclose(fh);
        return;
    end proc;

    tikzfile := module()
    option object;
    export getpreamb,setpreamb,getpostamb,setpostamb,reset,precode,postcode;

        getpreamb::static := proc()

        end proc;
        setpreamb::static := proc(s)

        end proc;
        getpostamb::static := proc()

        end proc;
        setpostamb::static := proc(s)

        end proc;
        tofile::static := proc(s,fn)
            ans := s;
            ans := cat(preamb,s,postamb);
            if(nargs=1) then
                return ans;
            else
                error;
            end if;
        end proc;
        reset::static := proc()

            fprintf(fh,"\\documentclass{standalone}\n");
            fprintf(fh,"\usepackage{xcolor}\n");
            fprintf(fh,"\\usepackage{tikz}\n");
            fprintf(fh,"\\usepackage{pgfplots}\n");
            fprintf(fh,"\\usepackage{pgfplots}\n");
            fprintf(fh,"\\usepgfplotslibrary{colormaps}\n");
            fprintf(fh,"\\pgfplotsset{compat=1.16}\n");
            fprintf(fh,"\\usetikzlibrary{positioning, backgrounds}\n\n");
            fprintf(fh,"\\begin{document}\n\n");
            fprintf(fh,cat("\\begin{tikzpicture}[",args[3..nargs],"]\n\n"));
        end proc;
    end module;

    getproj := proc(U1,t)
        E1 := Vector([1,0,0],datatype=float[8]);
        E2 := Vector([0,1,0],datatype=float[8]);
        E3 := Vector([0,0,1],datatype=float[8]);
        V1 := Vector([U1[1],U1[2],U1[3]],datatype=float[8])/sqrt(add(x^2,x=U1));
        U2 := E2-V1[2]*V1;
        V2 := U2/sqrt(add(x^2,x=U2));
        U3 := E3-V1[3]*V1-V2[3]*V2;
        V3 := U3/sqrt(add(x^2,x=U3));
        return Matrix([cos(t)*V2+sin(t)*V3,-sin(t)*V2+cos(t)*V3],datatype=float[8]);
    end proc;

    tikzhist := proc(V,rng)
        tk := module()
        option object;
        export V,N,tofile,out,color,rng;
            out::static := proc()
                ans := "";
                for i from 1 to N do

                end do;
                for sig in sigl do
                    c := X:-f(sig);
                    if(c>c1) then
                        next;
                    end if;
                    k := max(0,round(1000*(c-c0)/(c1-c0)));
                    k := 1000-k;
                    col := nprintf(cat("color of colormap={%d of ",color,"}"),k);
                    if(nops(sig)=1) then
                        p1 := verts(sig[1]);
                        ans := cat(ans,nprintf(cat("\\filldraw[",col,"] (%f,%f) circle (%dpt);\n"),op(p1),nodethickness));
                    elif(nops(sig)=2) then
                        p1,p2 := verts(sig[1]),verts(sig[2]);
                        ans := cat(ans,nprintf(cat("\\draw[",col,",",edgethickness,"] (%f,%f)--(%f,%f);\n"),op(p1),op(p2)));
                    elif(nops(sig)=3) then
                        p1,p2,p3 := verts(sig[1]),verts(sig[2]),verts(sig[3]);
                        ans := cat(ans,nprintf(cat("\\filldraw[",col,"] (%f,%f)--(%f,%f)--(%f,%f);\n"),op(p1),op(p2),op(p3)));
                    end if;
                end do;
                return nprintf(ans);
            end proc;
            tofile::static := proc(fn)
                savetikz(fn,out());
            end proc;
            out::static := proc()

            end proc;
        end module;
        return tk;
    end proc;

    tikzcloud := module()
    option object;
    export getcode,pointsize,pointunit,`?[]`,reset;
    local ModulePrint,ModuleApply;
        ModulePrint::static := proc()
            return nprintf("tikz scatter plot");
        end proc;
        getcode::static := proc(A,cols)
            N,m := Dimension(A);
            code := "";
            for i from 1 to N do
                if(nargs=2) then
                    col := [cols[i][]];
                    col := map(x->round(255*x),col);
                else
                    col := [0,0,0];
                end if;
                x,y := A[i,1],A[i,2];
                code1 := nprintf(cat("\\filldraw[color={rgb,%d:red,%d; green,%d; blue,%d}] (%f,%f) circle (%f",pointunit,");\n"),255,op(col),x,y,pointsize);
                code := cat(code,code1);
            end do;
            if(type(procname,indexed)) then
                fn := op(procname);
                return savetikz(fn,code);
            else
                return code;
            end if;
        end proc;
        ModuleApply::static := getcode;
        `?[]`::static := proc()
            return getcode[op(args[2])];
        end proc;
        getcolor::static := proc()
            return cmap;
        end proc;
        setcolor::static := proc(col)
            cmap := colormap(col);
            return;
        end proc;
        reset::static := proc()
            pointsize := .25;
            pointunit := "pt";
            setcolor('black');
        end proc;
        reset();
    end module;

end module;
