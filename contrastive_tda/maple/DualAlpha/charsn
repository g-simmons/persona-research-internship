getsn := proc(n)
uses combinat;
option remember;
    sig := [seq(i,i=1..n)];
    ans := [];
    while(true) do
        if(sig='FAIL') then
            break;
        end if;
        ans := [op(ans),sig];
        sig := nextperm(sig);
    end do;
    return ans;
end proc;

indperm := proc(sig)
    n := nops(sig);
    sigl := getsn(n);
    for i from 1 to n! do
        if(sigl[i]=sig) then
            return i;
        end if;
    end do;
    error;
end proc;

invperm := proc(sig)
    n := nops(sig);
    ans := [seq(0,i=1..n)];
    for i from 1 to n do
        ans[sig[i]] := i;
    end do;
    return ans;
end proc;

cyctype := proc(sig)
uses GroupTheory;
    n := nops(sig);
    mu := PermCycleType(Perm(sig));
    l := convert(mu,`+`);
    return [seq(1,i=1..n-l),op(mu)];
end proc;

sgnperm := proc(sig)
    mu := cyctype(sig);
    return mul((-1)^(m-1),m=mu);
end proc;

indperm := proc(sig)
    n := nops(sig);
    sigl := getsn(n);
    for i from 1 to n! do
        if(sigl[i]=sig) then
            return i;
        end if;
    end do;
    error;
end proc;

permrep1 := proc(sig,A)
    n := nops(sig);
    for i from 1 to n do
        for j from 1 to n do
            if(sig[j]=i) then
                A[i,j] := 1;
            end if;
        end do;
    end do;
    return A;
end proc;

permrep := proc(sig)
    n := nops(sig);
    A := Matrix(n,n,datatype=integer[4]);
    permrep1(sig,A);
    return A;
end proc;

regrep1 := proc(sig,A)
    n := nops(sig);
    sigl := getsn(n);
    m := n!;
    for i from 1 to m do
        for j from 1 to m do
            A[i,j] := 0;
            sig1 := sigl[i];
            sig2 := sigl[j];
            if(sig[sig2]=sig1) then
                A[i,j] := 1;
            end if;
        end do;
    end do;
    return;
end proc;

regrep := proc(sig)
    n := nops(sig);
    sigl := getsn(n);
    A := Matrix(n!,n!,datatype=integer[4]);
    regrep1(sig,A);
    return A;
end proc;

regrep1 := proc(sig,A)
    n := nops(sig);
    sigl := getsn(n);
    m := n!;
    for i from 1 to m do
        for j from 1 to m do
            A[i,j] := 0;
            sig1 := sigl[i];
            sig2 := sigl[j];
            if(sig[sig2]=sig1) then
                A[i,j] := 1;
            end if;
        end do;
    end do;
    return;
end proc;

clifton0 := proc(sig,T1,T2)
    n := nops(sig);
    T := acttab(sig,T2);
    ql := getgv(T1);
    for q in ql do
        if(map(sort,acttab(q,T))=T1) then
            return sgnperm(q);
        end if;
    end do;
    return 0;
end proc;

clifton1 := proc(sig,A)
    mu := op(procname);
    n := nops(sig);
    Tl := getst(mu);
    d := nops(Tl);
    for i from 1 to d do
        T1 := Tl[i];
        for j from 1 to d do
            T2 := Tl[j];
            A[i,j] := clifton0(sig,T1,T2);
        end do;
    end do;
end proc;

clifton := proc(sig)
    mu := op(procname);
    d := dimrep(mu);
    A := Matrix(d,d,datatype=integer[4]);
    clifton1[mu](sig,A);
    return A;
end proc;

irrep := proc(sig)
    mu := op(procname);
    n := nops(sig);
    A1 := clifton[mu]([seq(i,i=1..n)]);
    A := clifton[mu](sig);
    return A1^(-1).A;
end proc;

totab := proc(sig,mu)
    ans := [];
    k := 0;
    for m in mu do
        ans := [op(ans),sig[k+1..k+m]];
        k := k+m;
    end do;
    return ans;
end proc;

getst0 := proc(T)
    mu := map(nops,T);
    l := nops(mu);
    for i from 1 to l do
        if(T[i]<>sort(T[i])) then
            return false;
        end if;
    end do;
    for i from 2 to l do
        for j from 1 to mu[i] do
            if(T[i-1,j]>=T[i,j]) then
                return false;
            end if;
        end do;
    end do;
    return true;
end proc;

getgh := proc(T)
    mu := map(nops,T);
    n := convert(mu,`+`);
    sigl := getsn(n);
    ans := [];
    for sig in sigl do
        T1 := map(sort,acttab(sig,T));
        if(T1=T) then
            ans := [op(ans),sig];
        end if;
    end do;
    return ans;
end proc;

getgv := proc(T)
    return getgh(transtab(T));
end proc;

assertsf := proc()
    if(not type(Par,'procedure')) then
        error "load the SF package by John Stembridge";
    end if;
end proc;

transtab := proc(T)
    assertsf();
    mu := map(nops,T);
    nu := conjugate(mu);
    n := convert(mu,`+`);
    l := nops(nu);
    ans := [];
    for i from 1 to l do
        ans := [op(ans),[seq(T[j][i],j=1..nu[i])]];
    end do;
    return ans;
end proc;

getst := proc(la)
option remember;
    n := convert(la,`+`);
    sigl := getsn(n);
    ans := [];
    for sig in sigl do
        T := totab(sig,la);
        if(getst0(T)) then
            ans := [op(ans),T];
        end if;
    end do;
    return ans;
end proc;

acttab := proc(sig,T)
    mu := map(nops,T);
    return totab(sig[map(op,T)],mu);
end proc;

getpar := proc(n)
    assertsf();
    return Par(n);
end proc;

#character table of sn
chimu := proc(mu,sig)
    assertsf();
    nu := cyctype(sig);
    return scalar(s[op(mu)],mul(cat(p,k),k=nu));
end proc;

dimrep := proc(mu)
    n := convert(mu,`+`);
    return chimu(mu,[seq(i,i=1..n)]);
end proc;

projperm := proc(mu)
    n := convert(mu,`+`);
    A := Matrix(n!,n!);
    sigl := getsn(n);
    ans := Matrix(n!,n!);
    for i from 1 to n! do
        sig := sigl[i];
        A := regrep(sig);
        ans := ans+chimu(mu,sig)*A;
    end do;
    return ans*dimrep(mu)/n!;
end proc;

projreg := proc(mu)
    n := convert(mu,`+`);
    A := Matrix(n!,n!);
    sigl := getsn(n);
    ans := Matrix(n!,n!);
    for i from 1 to n! do
        sig := sigl[i];
        A := regrep(sig);
        ans := ans+chimu(mu,sig)*A;
    end do;
    return ans*dimrep(mu)/n!;
end proc;


