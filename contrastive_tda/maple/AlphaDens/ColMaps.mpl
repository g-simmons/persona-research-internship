ColMaps := module()
option package;
export colortab,colorvec,colormap,revcolor,tikzcolor,drawmap,imagemat,drawmat,heatmap,colmat,drawrgb,pngmat,heatmat,heatcolor,discmap,solidmap;
local maxfinite,heatcolor0;

    colormap := proc(colvec)
        if(whattype(colvec)='ColMap') then
            return colvec;
        elif(not type(args[1],'Vector')) then
            return colormap(colorvec(args));
        end if;
        md := module()
        option object;
        export `?[]`,`whattype`,colvec,getcol,colmap,col1,col2,init,alphamap,getalpha;
        local ModulePrint;
            ModulePrint::static := proc()
                print(seq(getcolor(250*i),i=0..4));
                return nprintf("color map");
            end proc;
            ModuleApply::static := proc(x)
                return colmap(x);
            end proc;
            `whattype`::static := proc()
                return 'ColMap';
            end proc;
            `?[]`::static := proc()
                return getcolor(op(args[2]));
            end proc;
            colmap := proc(x)
                i :=  round(1000*x);
                return getcolor(i);
            end proc;
            getcolor::static := proc(i)
                if(i<0) then
                    return col1;
                elif(i>1000) then
                    return col2;
                end if;
                return colvec[i+1];
            end proc;
            getalpha := proc(i)
                if(i>=0 and i<=1000) then
                    return 0.0;
                else
                    return 1.0;
                end if;
            end proc;
            alphamap::static := proc(x)
                i := round(1000*x);
                return getalpha(i);
            end proc;
            init::static := proc(V)
                colvec := Vector(V);
                col1 := Color([255,255,255]);
                col2 := Color([255,255,255]);
            end proc;
        end module;
        md:-init(colvec);
        return md;
    end proc;

    revcolor := proc(cmap)
        if(whattype(cmap)<>ColMap) then
            return revcolor(colormap(cmap));
        end if;
        colvec := cmap:-colvec;
        ans := colormap(Vector([seq(colvec[1001-i+1],i=1..1001)]));
        ans:-col1 := cmap:-col1;
        ans:-col2 := cmap:-col2;
        return ans;
    end proc;

#create the tikz code to generate the color using rng as endpoints
    tikzcolor := proc(cmap,cname,rng)
        if(type(args[3],'numeric')) then
            l := args[3];
            rng1 := [seq(round(1000*(i-1)/(l-1),i=1..l))];
            return procname(cmap,cname,rng1);
        end if;
        ans := "\\pgfplotsset{\n";
        ans := cat(ans,"colormap/",cmname,"/.style={/pgfplots/colormap={",cname,"}{\n");
        l := nops(rng);
        for i in rng do
            ans1 := nprintf("rgb=(%f,%f,%f)\n",op(getcolor(i)[1..3]));
            ans := cat(ans,ans1);
        end do;
        ans := cat(ans,"},\n},\n}\n");
        return nprintf(ans);
    end proc;

    colortab := proc(s)
    uses StringTools;
    local x;
        cmname := LowerCase(convert(s,'string'));
        tab := table();
        if(cmname="viridis") then
            tab[1000] := [253,231,37];
            tab[750] := [94,201,98];
            tab[500] := [33,145,140];
            tab[350] := [59,82,139];
            tab[0] := [68,1,84];
        elif(cmname="viridisa") then
            tab[0] := [253,231,37];
            tab[250] := [94,201,98];
            tab[500] := [33,145,140];
            tab[650] := [59,82,139];
            tab[1000] := [68,1,84];
        elif(cmname="virdiv") then
            tab[1000] := [253,231,37];
            tab[750] := [94,201,98];
            tab[500] := [33,145,140];
            tab[350] := [59,82,139];
            tab[0] := [68,1,84];
        elif(cmname="species1") then
            tab[0] := [0,110,29];
            tab[250] := [77,172,115];
            tab[500] := [179,220,195];
            tab[750] := [255,255,204];
            tab[1000] := [255,204,128];
        elif(cmname="grayblack") then
            tab[0] := [230,230,230];
            tab[1000] := [0,0,0];
        elif(cmname="blackbody") then
            tab[1000] := [255,255,255];
            tab[890] := [230,230,53];
            tab[580] := [227,105,5];
            tab[390] := [178,34,34];
            tab[0] := [0,0,0];
        elif(cmname="species2") then
            tab[1000] := map(round,[1.0,.8,.502]*255);
            tab[800] := map(round,[1.0,1.0,.8]*255);
            tab[600] := map(round,[.702,.863,.765]*255);
            tab[400] := map(round,[.302,.6745,.451]*255);
            tab[200] := map(round,[0.0,.431,.114]*255);
            tab[0] := map(round,[0.0,.3,.05]*255);
        elif(cmname="plasma") then
            tab[1000] := [240,249,33];
            tab[860] := [254,188,43];
            tab[710] := [244,136,73];
            tab[570] := [219,92,104];
            tab[430] := [185,50,137];
            tab[290] := [139,10,165];
            tab[140] := [84,2,163];
            tab[0] := [13,8,135];x
        elif(cmname="bioheat") then
            tab[0] := [25,74,0];
            tab[167] := [106,141,21];
            tab[333] := [152,182,24];
            tab[500] := [225,235,93];
            tab[667] := [243,172,51];
            tab[833] := [239,127,40];
            tab[1000] := [223,58,27];
        elif(cmname="bioheata") then
            tab[1000] := [25,74,0];
            tab[833] := [106,141,21];
            tab[677] := [152,182,24];
            tab[500] := [225,235,93];
            tab[333] := [243,172,51];
            tab[167] := [239,127,40];
            tab[0] := [223,58,27];
        elif(cmname="ising") then
            tab[0] := [101,127,214];
            tab[167] := [125,139,200];
            tab[333] := [148,145,174];
            tab[500] := [171,152,145];
            tab[667] := [193,162,108];
            tab[833] := [222,167,77];
            tab[1000] := [234,177,61];
        elif(cmname="seq1") then
            tab[0] := [214,238,245];
            tab[250] := [157,195,210];
            tab[500] := [115,167,185];
            tab[750] := [71,135,158];
            tab[1000] := [34,110,135];
        elif(cmname="div1") then
            tab[0] := [67,139,130];
            tab[250] := [167,194,164];
            tab[500] := [241,236,216];
            tab[750] := [231,168,119];
            tab[1000] := [195,94,52];
        elif(cmname="div2") then
            tab[0] := map(round,[.23,.299,.754]*255);
            tab[500] := map(round,[.865,.865,.865]*255);
            tab[1000] := map(round,[.706,.334,.046]*255);
        elif(cmname="bw") then
            tab[0] := [0,0,0];
            tab[1000] := [255,255,255];
        elif(cmname="wb") then
            tab[0] := [255,255,255];
            tab[1000] := [0,0,0];
        elif(cmname="bb") then
            tab[0] := [0,0,0];
            tab[1000] := [0,0,0];
        else
            c := map(round,[Color(s)[]]*255);
            tab[0] := c;
            tab[1000] := c;
        end if;
        return tab;
    end proc;

    colorvec := proc(tab)
        if(not type(args[1],'table')) then
            return colorvec(colortab(args));
        end if;
        il := sort(map(op,[indices(tab)]));
        if(min(il)<>0 or max(il)<>1000) then
            error "must have endpoints of 0,1000";
        end if;
        l := nops(il);
        ans := Vector(1001);
        ans[1] := Color(tab[0]);
        for j from 2 to l do
            i1,i2 := il[j-1],il[j];
            col0,col1 := Color(tab[i1]),Color(tab[i2]);
            for i from i1+1 to i2 do
                t := evalf((i-i1)/(i2-i1));
                ans[i+1] := Color((1-t)*col0[1..3]+t*col1[1..3]);
            end do;
        end do;
        return ans;
    end proc;

    imagemat := proc(f,rng,N:=500)
    local V;
        try(f([0,0]))
        catch: "invalid input";
            return procname(V->f(V[1],V[2]),args[2..nargs]);
        end try;
        x0,x1 := op(rng[1]);
        y0,y1 := op(rng[2]);
        if(type(args[3],'numeric')) then
            t := min((x1-x0)/N,(y1-y0)/N);
            m := ceil((x1-x0)/t);
            n := ceil((y1-y0)/t);
        elif(type(args[3],'list')) then
            m,n := op(args[3]);
        end if;
        B := allocla[float[8]]([n,m]);
        for i from 1 to m do
            for j from 1 to n do
                x := (x0+(x1-x0)*(i-.5)/m);
                y := (y0+(y1-y0)*(j-.5)/n);
                B[n-j+1,i] := f([x,y]);
            end do;
        end do;
        return B;
    end proc;

    imagemat := proc(f,rng,N:=500)
    local V;
        try(f([0,0]))
        catch: "invalid input";
            return procname(V->f(V[1],V[2]),args[2..nargs]);
        end try;
        x0,x1 := op(rng[1]);
        y0,y1 := op(rng[2]);
        if(type(args[3],'numeric')) then
            t := min((x1-x0)/N,(y1-y0)/N);
            m := ceil((x1-x0)/t);
            n := ceil((y1-y0)/t);
        elif(type(args[3],'list')) then
            m,n := op(args[3]);
        end if;
        B := allocla[float[8]]([n,m]);
        for i from 1 to m do
            for j from 1 to n do
                x := (x0+(x1-x0)*(i-.5)/m);
                y := (y0+(y1-y0)*(j-.5)/n);
                B[n-j+1,i] := f([x,y]);
            end do;
        end do;
        return B;
    end proc;

    colmat := proc(B,col:='viridis')
        cmap := colormap(col);
        m,n := Dimension(B);
        arr := Array(seq(1..k,k=[m,n,3]),datatype=float[8]);
        for i from 1 to m do
            for j from 1 to n do
                arr[i,j,1],arr[i,j,2],arr[i,j,3] := cmap(B[i,j])[];
            end do;
        end do;
        return arr;
    end proc;

    heatmat := proc(B,r)
        if(nargs=1 or args[2]=false) then
            return heatmat(B,0..1);
        end if;
        if(r=true) then
            a0,a1 := min(B),maxfinite(B);
        elif(type(r,'numeric')) then
            a0,a1 := min(B),r;
        elif(type(r,`..`)) then
            a0,a1 := op(r);
        end if;
        cmap := heatcolor();
        m,n := Dimension(B);
        arr := Array(seq(1..k,k=[m,n,4]),datatype=float[8]);
        for i from 1 to m do
            for j from 1 to n do
                c := (B[i,j]-a0)/(a1-a0);
                arr[i,j,1],arr[i,j,2],arr[i,j,3] := cmap(c)[];
                arr[i,j,4] := cmap:-getalpha(c);
            end do;
        end do;
        Embed(Create(arr));
        return arr;
    end proc;

    heatmat := module()
    option object;
    export getcolor,setcolor,draw,cmap;
    local ModuleApply,ModulePrint;
        ModulePrint::static := proc()
            print(cmap);
            return nprintf("draws a heat map");
        end proc;
        draw::static := proc(B,r)
            m,n := Dimension(B);
            if(nargs=1 or r=true) then
                a0,a1 := min(B),maxfinite(B);
            elif(type(r,'numeric')) then
                a0,a1 := min(B),r;
            elif(type(r,`..`)) then
                a0,a1 := op(r);
            end if;
            arr := Array(seq(1..k,k=[m,n,4]),datatype=float[8]);
            for i from 1 to m do
                for j from 1 to n do
                    c := (B[i,j]-a0)/(a1-a0);
                    arr[i,j,1],arr[i,j,2],arr[i,j,3] := cmap(c)[];
                    arr[i,j,4] := cmap:-getalpha(c);
                end do;
            end do;
            Embed(Create(arr));
            return arr;
        end proc;
        ModuleApply::static := draw;
        if(nargs=1 or args[2]=false) then
            return heatmat(B,0..1);
        end if;
        getcolor::static := proc()
            return cmap;
        end proc;
        setcolor::static := proc(col)
            cmap := colormap(col);
        end proc;
        setcolor('viridis');
    end module;

    discmap0 := proc(cl)
        k := nops(cl);
        colvec := Vector(1001);
        for i from 1 to 1001 do
            i1 := floor((i-1)*k/1001)+1;
            colvec[i] := cl[i1];
        end do;
        return colormap(colvec);
    end proc;

    discmap := proc(col,k)
        if(type(args[1],'list')) then
            return discmap0(args);
        end if;
        cmap := colormap(col);
        colvec0 := cmap:-colvec;
        colvec := Vector(1001);
        for i from 0 to 1000 do
            i1 := floor(1000/(k-1)*floor(k*i/1001))+1;
            colvec[i+1] := colvec0[i1];
        end do;
        return colormap(colvec);
    end proc;

    solidmap := proc(col)
        c := Color(col);
        return colormap(Vector([seq(c,i=1..1001)]));
    end proc;

    drawim := proc(arr)
    uses ImageTools;
        return Embed(Create(arr));
    end proc;

    drawmat := proc(B,col:='viridis')
    uses ImageTools;
        arr := colmat(B,col);
        im := Create(arr);
        return Embed(im);
    end proc;

    drawrgb := proc(arr)
        return Embed(Create(arr));
    end proc;

    drawmap := proc(f,rng,N:=500,col:='viridis')
    local x;
        A := imagemat(f,rng,N);
        if(type(procname,indexed) and op(procname)=true) then
            a := min(A);
            b := maxfinite(A);
            map[inplace](x->(x-a)/(b-a),A);
        end if;
        return drawmat(A,col);
    end proc;

    maxfinite := proc(A)
        ans := -Float(infinity);
        for c in A do
            if(c<>Float(infinity)) then
                ans := max(ans,c);
            end if;
        end do;
        return ans;
    end proc;

    heatcolor0 := colormap('viridis');

    heatcolor := proc(col)
        if(nargs=0) then
            return heatcolor0;
        elif(nargs=1) then
            heatcolor0 := colormap(col);
            return;
        end if;
    end proc;

    heatmap := proc(im,cmap:=heatcolor())
        print(hi,cmap);
        cmap1 := colormap(cmap);
        A := convert(im,'Matrix');
        m1,m2 := Dimension(A);
        c0 := min(A);
        if(nargs=3) then
            c1 := args[3];
        else
            c1 := maxfinite(A);
        end if;
        arr := Array(1..m1,1..m2,1..3);
        for i1 from 1 to m1 do
            for i2 from 1 to m2 do
                c := im[i1,i2];
                x := (c-c0)/(c1-c0);
                arr[i1,i2,1],arr[i1,i2,2],arr[i1,i2,3] := cmap1(x)[];
            end do;
        end do;
        img := Create(arr);
        Embed(img);
        return img;
    end proc;

    heatmap := heatmat;

end module;
