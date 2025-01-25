im2mat := proc(im)
    m,n,k := arrdim(im);
    A := matf(m,n);
    for i from 1 to m do
        for j from 1 to n do
            if(im[i,j,1]<.5 and im[i,j,4]<.5) then
                A[i,j] := 1.0;
            end if;
        end do;
    end do;
    return A;
end proc;

sampmat := proc(A,N,rng:=[-1..1,-1..1])
    m,n := Dimension(A);
    V := arr2vec(A);
    J := veci(sample(V,N));
    f,g := ordmaps([m,n]);
    a1,b1 := op(rng[1]);
    a2,b2 := op(rng[2]);
    ans := matf(N,2);
    for k from 1 to N do
        i,j := op(f(J[k]));
        ans[k,1] := evalf(a1+(j-randf(0,1))/n*(b1-a1));
        ans[k,2] := evalf(a2+(m-i+randf(0,1))/m*(b2-a2));
    end do;
    return ans;
end proc;

getbdy := proc(A)
    m,n := Dimension(A);
    ans := matf(m,n);
    for i from 2 to m-1 do
        for j from 2 to n-1 do
            vals := [A[i-1,j-1],A[i-1,j],A[i-1,j+1],A[i,j-1],A[i,j],A[i,j+1],A[i+1,j-1],A[i+1,j],A[i+1,j+1]];
            ans[i,j] := max(vals)-min(vals);
        end do;
    end do;
    return ans;
end proc;
