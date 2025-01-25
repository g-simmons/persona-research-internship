#alpha shapes of kernel density estimators
AlphaDens := module()
option package;
export gaussfit,densrange,drawdens,maxgauss,sampshape,shapecolor,densalpha,densland,densplex;
local sortcut;

    gaussfit0 := proc(k)
    option remember;
    uses LinAlg;
        m := 2^k;
        return allocla[float[8]](m,m);
    end proc;

    gaussfit1 := proc(U::Array(datatype=float[8]),A1::Array(datatype=float[8]),Z::Array(datatype=float[8]),k::integer[4],N::integer[4],m::integer[4])
        b := 0.0;
        for i from 1 to N do
            b := b+U[i];
        end do;
        for j from 1 to m do
            a := 0.0;
            for i from 1 to N do
                a := a+U[i]*A1[i,j];
            end do;
            Z[k,j] := a/b;
        end do;
    end proc;

    gaussfit1 := Compiler:-Compile(gaussfit1);

#see the paper.
    gaussfit := proc(f,B,A1:=f:-A)
    uses LinAlg;
        N,m := f:-N,f:-m;
        M := Dimension(B)[1];
        d := Dimension(A1)[2];
        if(not type(procname,indexed)) then
            Z,bb := allocla[float[8]]([M,d],M);
            gaussfit[Z,bb](args);
            return Z,bb;
        end if;
        Z,bb := op(procname);
        yy,zz := gaussfit0(ceil(log(m)/log(2)));
        for k from 1 to M do
            getrow1(B,k,yy,m);
            f:-setpoint(yy);
            c := f:-getdens1();
            f:-getmean1(zz);
            bb[k] := c/f:-rho(zz,yy);
            gaussfit1(f:-U,A1,Z,k,N,d);
        end do;
    end proc;

    densrange := proc(f,mindens,N:=10000)
    local c;
        B := f:-sample(N);
        A,bb := gaussfit(f,B);
        N1,m := Dimension(A);
        h := f:-h;
        pow := map(c->2*h^2*log(c),bb);
        a1 := -2*h^2*log(mindens);
        rng := [];
        for j from 1 to m do
            x0 := min(seq(A[k,j]-sqrt(max(a1+pow[k],0.0)),k=1..N1));
            x1 := max(seq(A[k,j]+sqrt(max(a1+pow[k],0.0)),k=1..N1));
            rng := [op(rng),x0..x1];
        end do;
        return rng;
    end proc;

    drawdens := proc(f,mindens,rng,N)
    local c;
        if(type(procname,indexed)) then
            return drawdens(args,op(procname));
        elif(nargs=2) then
            return drawdens(args,1000);
        elif(nargs=3) then
            return drawdens(f,mindens,10000,args[3]);
        elif(type(rng,'numeric')) then
            printf("computing the range...\n");
            return procname(f,mindens,densrange(f,mindens,rng),N);
        end if;
        im := imregion(rng,N);
        printf("evaluating the density...\n");
        im:-setmap(f);
        h := f:-h;
        map[inplace](c->-2*h^2*log(c),im:-vec);
        c0 := min(im:-vec);
        a1 := -2*h^2*log(mindens);
        map[inplace](c->(c-c0)/(a1-c0),im:-vec);
        return heatmap(convert(im,'Matrix'),denscolor());
    end proc;

    maxgauss0 := proc(S::Array(datatype=float[8]),pow::Array(datatype=float[8]),V::Array(datatype=float[8]),n::integer[4],m::integer[4])
        ind := 0;
        ans := Float(infinity);
        for k from 1 to n do
            r := -pow[k];
            for j from 1 to m do
                c := S[k,j]-V[j];
                r := r+c*c;
            end do;
            if(r<ans) then
                ind := k;
                ans := r;
            end if;
        end do;
        return ind;
    end proc;

    maxgauss0 := Compiler:-Compile(maxgauss0);

#max of gaussians object. finds the index or indices of the cells of a
#given set of points, or can apply the function to a given point.
    maxgauss := proc(S,aa,h)
        md := module()
        option object;
        export S,aa,h,n,m,pow,init,getsize,`numelems`,getweight,getweight1,getval,getval1,indval,indval1,indweight,indweight1,getcell,getcell1,getcells,drawheat,colmap;
        local ModuleApply,ModulePrint,V0;
            ModulePrint::static := proc()
            local s;
                s := "max of weighted gaussians, %d sites in R^%d";
                return nprintf(s,n,m);
            end proc;
            getcells::static := proc(B,J)
                N1 := Dimension(B)[1];
                if(nargs=1) then
                    J1 := allocla[integer[4]](N1);
                    return procname(B,J1);
                end if;
                for k from 1 to N1 do
                    getrow(B,k,V0,m);
                    J[k] := getcell1(V0);
                end do;
                return J;
            end proc;
            getcell::static := proc(x)
                setvec(V0,x);
                return getcell1(V0);
            end proc;
            getcell1::static := proc(V)
                return maxgauss0(S,pow,V,n,m);
            end proc;
            getval::static := proc(x)
                setvec(V0,x);
                if(type(procame,indexed)) then
                    return indval1(op(procname),V0);
                else
                    return getval1(V0);
                end if;
            end proc;
            getval1::static := proc(V)
                return exp(-getweight1(V)/2/h^2);
            end proc;
            getweight::static := proc(x)
                setvec(V0,x);
                if(type(procame,indexed)) then
                    return indweight1(op(procname),V0);
                else
                    return getweight1(V0);
                end if;
            end proc;
            getweight1::static := proc(V)
                return indweight1(getcell1(V),V);
            end proc;
            indval::static := proc(i,x)
                setvec(V0,x);
                return indval1(i,V0);
            end proc;
            indval1::static := proc(i,V)
                return exp(-indweight1(i,V));
            end proc;
            indweight::static := proc(i,x)
                setvec(V0,x);
                return indweight1(i,V0);
            end proc;
            indweight1::static := proc(i,V)
                return add((S[i,j]-V[j])^2,j=1..m)-pow[i];
            end proc;
            ModuleApply::static := getval;
            getsize::static := proc()
                return n;
            end proc;
            `numelems`::static := proc()
                return getsize();
            end proc;
            drawheat::static := proc(mindens,scale:=1.0,M:=1000)
                S1 := S.scale;
                maxdens := max(aa);
                a1 := -2*h^2*log(mindens);
                return powheat[M,args[5..nargs]](S1,pow,a1);
            end proc;
            init::static := proc()
            local x;
                S,aa,h := args;
                n,m := Dimension(S);
                pow := vecf([seq(2*h^2*log(aa[k]),k=1..n)]);
                V0 := allocla[float[8]](m);
            end proc;
        end module;
        md:-init(S,aa,h);
        return md;
    end proc;

#returns the result of sorting aa in reverse order, discarding
#trailing elements below mindens.
    sortcut := proc(aa,mindens)
        J := sortinds(aa,`>`);
        N := Dimension(aa);
        for n from 1 to N do
            if(aa[J[n]]<mindens) then
                break;
            end if;
        end do;
        n := n-1;
        return [seq(J[i],i=1..n)];
    end proc;

#sampshape0(S,B,aa,pow,S1,B1,aa1,ord,h,mindens,s,J,n,m,M);
    densland0 := proc(S::Array(datatype=float[8]),eps::float[8],ord::Array(datatype=integer[4]),J::Array(datatype=integer[4]),N::integer[4],m::integer[4])
        n := 0;
        for k from 1 to N do
            k0 := ord[k];
            for i from 1 to n do
                k1 := J[i];
                r := 0.0;
                for j from 1 to m do
                    x := S[k1,j]-S[k0,j];
                    r := r+x*x;
                end do;
                if(r<eps) then
                    break;
                end if;
            end do;
            if(i<n+1) then
                next;
            end if;
            n := n+1;
            J[n] := k0;
        end do;
        return n;
    end proc;

    densland0 := Compiler:-Compile(densland0);

    densland := proc(S,aa,eps,mindens)
        N,m := Dimension(S);
        ord := veci(sortcut(aa,mindens));
        N1 := Dimension(ord);
        J := veci(N1);
        n := densland0(S,eps,ord,J,N1,m);
        return J[1..n];
    end proc;

#sample from the alpha shape of f. N1 can be the number of samples
#taken from f, or a given matrix of points from which to apply the
#gaussfit procedure. s determines the minimum spacing between sites.
    sampshape := proc(f,N1,s,mindens:=0)
        if(not type(procname,indexed) or op(procname)<>true) then
            return sampshape[true](args)[1..2];
        end if;
        h := f:-h;
        if(nargs>2) then
            S,aa,B := sampshape[true](f,N1);
            eps := -2*h^2*log(s);
            inds := convert(densland(S,aa,eps,mindens),'list');
            return S[inds],aa[inds],B[inds];
        end if;
        if(type(N1,'Matrix')) then
            B := N1;
        else
            B := f:-sample(N1);
        end if;
        S,aa := gaussfit(f,B);
        return S,aa,B;
    end proc;

#compute the density complex.
    densplex := proc(g,mindens,s,d)
        if(not type(procname,indexed) or op(procname)=false) then
            return densplex[true](args)[1];
        elif(type(args[1],'Matrix')) then
            return procname(maxgauss(args[1..3]),args[4..nargs]);
        end if;
        h,S,pow := g:-h,g:-S,g:-pow;
        N := Dimension(S)[1];
        a1 := -2*h^2*log(mindens);
        eps := -2*h^2*log(s);
        X,Phi := alphaplex[true](S,pow,a1,d);
        Y := fplex(N);
        for k from 0 to d do
            for sig in X[k] do
                a := X:-f(sig);
                a := max(a,seq(eps-pow[i],i=sig));
                if(a<=a1) then
                    Y:-addfilt(sig,a);
                end if;
            end do;
        end do;
        return Y,Phi;
    end proc;


## densalpha: Construct Alpha Shape for Kernel Density Estimator
##
## This procedure creates an object representing the alpha shape of a
## kernel density estimator (KDE) or a set of points.
## collects everything you would want to store from a finite alpha
## shape. begins by sampling from the kde up to the minimum cutoff
## cdf. it can then compute the alpha complex with slack, draw the heat
## map or the complex, etc.
##
## Parameters:
## A      - Either a KDE object or a matrix of points
## h      - Bandwidth parameter for the Gaussian kernels
## N1     - Number of samples to take from the KDE (if A is a KDE),
##          or number of points to use (if A is a matrix)
## s      - Slack parameter controlling minimum spacing between sites
## mindens - Minimum density threshold for inclusion in the alpha shape
##
## Returns:
## A module object with methods for analyzing and visualizing the alpha shape,
## including:
## - drawheat: Draw a heat map of the density
## - drawalpha: Visualize the alpha shape
## - densplex: Compute the density complex (filtered simplicial complex)
## - powrange: Compute the range of power values
## - convland: Convert landmarks to a different coordinate system
##
## This function serves as the main interface for creating and working with
## alpha shapes of density estimators in the AlphaDens module.

    densalpha := proc(A,h,N1,s,mindens)
        if(whattype(A)='KDE') then
            f := args[1];
            return densalpha(f:-A,f:-h,args[2..nargs]);
        end if;
        md := module()
        option object;
        export A,h,f,N,m,n,g,B,S,s,aa,powdata,powrange,getframe,convland,mindens,drawheat,drawalpha,densplex,init;
        local ModulePrint;
            ModulePrint::static := proc()
                return nprintf("alpha shape, %d sites in R^%d",n,m);
            end proc;
            drawheat::static := proc(Q:=getframe(),M:=1000)
                return g:-drawheat(mindens,Q,M);
            end proc;
            getframe::static := proc()
                if(m>2) then
                    return randframe(m,2);
                else
                    return Matrix(IdentityMatrix(m),datatype=float[8]);
                end if;
            end proc;
            densplex::static := proc(d)
                if(not type(procname,indexed) or op(procname)=false) then
                    return densplex[true](d)[1];
                end if;
                return AlphaDens:-densplex[true](g,mindens,s,d);
            end proc;
            convland::static := proc(A1)
                return gaussfit(f,B,A1)[1];
            end proc;
            powrange::static := proc()
                a0 := -2*h^2*log(max(aa));
                a1 := -2*h^2*log(mindens);
                return a0..a1;
            end proc;
            drawalpha::static := proc(d,A1)
                if(not type(procname,indexed)) then
                    return drawalpha[2000](args);
                end if;
                if(nargs=2) then
                    S1 := convland(A1);
                else
                    S1 := S.getframe();
                end if;
                pow,a1 := powdata()[2..3];
                if(type(d,'numeric')) then
                    X := getplex(d);
                else
                    X := d;
                end if;
                return DrawAlpha:-drawalpha(S1,pow,a1,X);
            end proc;
            powdata::static := proc()
                return S,g:-pow,-2*h^2*log(mindens);
            end proc;
            init::static := proc(A1,h1,N1,s1,mindens1)
                A,h,s,mindens := A1,h1,s1,mindens1;
                N,m := Dimension(A);
                f := getkde(A,h);
                S,aa,B := sampshape[true](f,N1,s,mindens);
                n := Dimension(S)[1];
                g := maxgauss(S,aa,h);
            end proc;
        end module;
        md:-init(A,h,N1,s,mindens);
        return md;
    end proc;

end module;
