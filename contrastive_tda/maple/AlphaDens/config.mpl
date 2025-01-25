sampconf := proc(N,h,nsteps,beta,R:=3)
local r,xx;
    H0 := r->4*(1/r^6-1/r^3);
    H := add(add(H0((cat(x,i)-cat(x,j))^2+(cat(y,i)-cat(y,j))^2),j=i+1..3),i=1..3);
    H := H+add(cat(x,i)^2+cat(y,i)^2,i=1..3)/2/R^2;
    vars := [x1,y1,x2,y2,x3,y3];
    params := [];
    met := metmodel([H,vars,params],beta);
    met:-g0:-setcov(R);
    ans := met:-sample(N,h,nsteps);
    ans := rowmap(centconf,ans);
    return ans;
end proc;

centconf := proc(V)
    p1,p2,p3 := [V[1],V[2]],[V[3],V[4]],[V[5],V[6]];
    if(type(procname,indexed) and nops(procname)=1) then
        i := op(procname);
        p0 := [p1,p2,p3][i];
    else
        p0 := (p1+p2+p3)/3;
    end if;
    return [op(p1-p0),op(p2-p0),op(p3-p0)];
end proc;

normconf := proc(V)
    p1,p2,p3 := [V[1],V[2]],[V[3],V[4]],[V[5],V[6]];
    p2 := p2-p1;
    p3 := p3-p1;
    p1 := p1-p1;
    r := sqrt(p2[1]^2+p2[2]^2);
    c,s := op(p2/r);
    p2 := [c*p2[1]+s*p2[2],-s*p2[1]+c*p2[2]];
    p3 := [c*p3[1]+s*p3[2],-s*p3[1]+c*p3[2]];
    r := sqrt(p2[1]^2+p2[2]^2);
    return [op(p1),op(p2),op(p3)]/r;
end proc;

conforbs := proc(B)
uses combinat;
    N,m := Dimension(B);
    if(m<>6) then
        error;
    end if;
    ans := matf(6*N,m);
    for i1 from 1 to N do
        k1 := 6*(i1-1)+1;
        for j from 1 to 6 do
            ans[k1,j] := B[i1,j];
        end do;
        sig := [1,2,3];
        for i2 from 1 to 5 do
            k2 := k1+i2;
            sig := nextperm(sig);
            for j from 1 to 3 do
                ans[k2,2*j-1] := ans[k1,2*sig[j]-1];
                ans[k2,2*j] := ans[k1,2*sig[j]];
            end do;
        end do;
    end do;
    return ans;
end proc;

drawconf := proc(V,ord:=true)
    if(type(V,'Matrix')) then
        m,n := Dimension(V);
        return display([seq(drawconf(V[i]),i=1..m)]);
    elif(type(V,'list')) then
        n := nops(V);
    elif(type(V,'Vector')) then
        n := Dimension(V);
    end if;
    n := n/2;
    if(ord) then
        cl := [blue,red,green,brown,seq(black,i=5..n)];
    else
        cl := [seq(black,i=1..n)];
    end if;
    ans := [];
    for i from 1 to n do
        p := [V[2*i-1],V[2*i]];
        for j from i+1 to n do
            q := [V[2*j-1],V[2*j]];
            r := sqrt((p[1]-q[1])^2+(p[2]-q[2])^2);
            if(r<1.3) then
                ans := [op(ans),line(p,q,linestyle=dash)];
            end if;
        end do;
    end do;
    ans := [op(ans),seq(point([V[2*i-1],V[2*i]],symbol=solidcircle,color=cl[i]),i=1..n)];
    return display(ans,view=[-2..2,-2..2]);
end proc;

tikzconf := proc(V,ord:=true)
    s := "\\begin{tikzpicture}\n";
    if(type(V,'Matrix')) then
        m,n := Dimension(V);
        for i from 1 to m do
            s := cat(s,tikzconf0(convert(V[i],'list'),ord));
        end do;
        return s;
    elif(type(V,'list')) then
        n := nops(V);
    elif(type(V,'Vector')) then
        n := Dimension(V);
    end if;
    s := cat(s,tikzconf0(convert(V,'list'),ord));
    s := cat(s,"\\end{tikzpicture}\n");
    printf(s);
end proc;

tikzconf0 := proc(xl,ord)
    n := nops(xl)/2;
    if(ord) then
        cl := ["blue","red","green","brown",seq("black",i=5..n)];
    else
        cl := [seq("black",i=1..n)];
    end if;
    s := "";
    for i from 1 to n do
        p := [xl[2*i-1],xl[2*i]];
        for j from i+1 to n do
            q := [xl[2*j-1],xl[2*j]];
            r := sqrt((p[1]-q[1])^2+(p[2]-q[2])^2);
            if(r<1.3) then
                s := cat(s,nprintf("\\draw[black,semithick,densely dashed](%f,%f)--(%f,%f);\n",xl[2*i-1],xl[2*i],xl[2*j-1],xl[2*j]));
            end if;
        end do;
    end do;
    for i from 1 to n do
        s := cat(s,nprintf(cat("\\filldraw[",cl[i],"] (%f,%f) circle (1.000000pt);\n"),xl[2*i-1],xl[2*i]));
    end do;
    return s;
end proc;

drawfiber := proc(B)
    N := Dimension(B)[1];
    ans := [];
    for i from 1 to N do
        p1 := [B[i,1],B[i,2]];
        p2 := [B[i,3],B[i,4]];
        p3 := [B[i,5],B[i,6]];
        p2 := p2-p1;
        p3 := p3-p1;
        p1 := p1-p1;
        r := sqrt(p2[1]^2+p2[2]^2);
        p2 := p2/r;
        p3 := p3/r;
        c,s := op(p2);
        p2 := [c*p2[1]+s*p2[2],-s*p2[1]+c*p2[2]];
        p3 := [c*p3[1]+s*p3[2],-s*p3[1]+c*p3[2]];
        ans := [op(ans),point(p1,symbol=solidcircle,symbolsize=4,color=blue)];
        ans := [op(ans),point(p2,symbol=solidcircle,symbolsize=4,color=red)];
        ans := [op(ans),point(p3,symbol=solidcircle,symbolsize=4,color=black)];
    end do;
    return display(ans,view=[-1..2,-2..2]);
end proc;

stereomap := proc(p)
    x,y := op(p);
    return [2*x,2*y,1-x^2-y^2]/(1+x^2+y^2);
end proc;

hopfmap := proc(vv)
    vv1,vv2 := vv[[1,3,5]],vv[[2,4,6]];
    q1 := 1/sqrt(2.0)*[1,0,-1];
    q2 := 1/sqrt(6.0)*[1,-2,1];
    ww := [vecdot(vv1,q1),vecdot(vv2,q1),vecdot(vv1,q2),vecdot(vv2,q2)];
    r := vecnorm(ww);
    ww := ww/r;
    z1,z2 := ww[1]+ww[2]*I,ww[3]+ww[4]*I;
    if(true) then
        z := z1/z2;
        p := [Re(z),Im(z)];
        ans := stereomap(p);
    else
        z := z2/z1;
        p := [Re(z),Im(z)];
        ans := stereomap(p);
        ans[2] := -ans[2];
        ans[3] := -ans[3];
    end if;
    return r*ans;
end proc;

