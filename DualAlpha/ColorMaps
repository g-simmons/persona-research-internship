ColorMaps := module()
option package;
export colorvec,colortab,colormap,imcmap,colpar;
local colormap0;

    colormap := proc(s)
        if(not type(args[1],'table')) then
            return colormap0(args);
        end if;
        argl := [s];
        md := module()
        option object;
        export `?[]`,tab,tab0,inds,getcolor,tikz,cmname,f,colmap,scale,col0,col1;
        local ModulePrint;
            ModulePrint::static := proc()
                print(seq(getcolor(250*i),i=0..4));
                return nprintf(cmname);
            end proc;
            ModuleApply::static := proc(x)
                return colmap(x);
            end proc;
            `?[]`::static := proc()
                return getcolor(op(args[2]));
            end proc;
            colmap := proc(x)
                y := f(x/scale);
                i :=  round(1000*y);
                return getcolor(i);
            end proc;
            getcolor::static := proc(i)
                if(i<0) then
                    return col0;
                elif(i>1000) then
                    return col1;
                end if;
                return tab[i];
            end proc;
            tikz::static := proc(l)
                ans := "\\pgfplotsset{\n";
                ans := cat(ans,"colormap/",cmname,"/.style={/pgfplots/colormap={",cmname,"}{\n");
                for i from 1 to l do
                    j := round(1000*(i-1)/(l-1));
                    ans1 := nprintf("rgb=(%f,%f,%f)\n",op(getcolor(j)[1..3]));
                    ans := cat(ans,ans1);
                end do;
                ans := cat(ans,"},\n},\n}\n");
                return nprintf(ans);
            end proc;
            tab0 := op(argl);
            tab := colortab(tab0);
            inds := map(op,[indices(tab0)]);
            col0,col1 := tab[0],tab[1000];
            cmname := "mycolor";
            f := colpar("div");
            scale := 1.0;
        end module;
        return md;
    end proc;

    colormap0 := proc(s)
    uses StringTools;
    local x;
        cmname := LowerCase(convert(s,'string'));
        tab := table();
        f := x->x/sqrt(1+x^2);
        if(cmname="viridis") then
            tab[1000] := [253,231,37];
            tab[750] := [94,201,98];
            tab[500] := [33,145,140];
            tab[350] := [59,82,139];
            tab[0] := [68,1,84];
        elif(cmname="virdiv") then
            tab[1000] := [253,231,37];
            tab[750] := [94,201,98];
            tab[500] := [33,145,140];
            tab[350] := [59,82,139];
            tab[0] := [68,1,84];
            f := colpar('div');
        elif(cmname="species1") then
            tab[0] := [0,110,29];
            tab[250] := [77,172,115];
            tab[500] := [179,220,195];
            tab[750] := [255,255,204];
            tab[1000] := [255,204,128];
        elif(cmname="species2") then
            tab[1000] := map(round,[1.0,.8,.502]*255);
            tab[800] := map(round,[1.0,1.0,.8]*255);
            tab[600] := map(round,[.702,.863,.765]*255);
            tab[400] := map(round,[.302,.6745,.451]*255);
            tab[200] := map(round,[0.0,.431,.114]*255);
            tab[0] := map(round,[0.0,.3,.05]*255);
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
            f := colpar('div');
        else
            error;
        end if;
        md := colormap(eval(tab));
        md:-f := f;
        if(nargs=2) then
            md:-f := colpar(args[2]);
        end if;
        md:-cmname := cmname;
        return md;
    end proc;

#color parametrization. allowable range maps into 0..1.
    colpar := proc(typ)
    local x;
        if(not type(typ,'string')) then
            return colpar(typeform(typ));
        elif(typ="div") then
            return x->(1+x/sqrt(1+x^2))/2;
        elif(typ="heat") then
            return x->x/sqrt(1+x^2);
        elif(typ="lin") then
            if(nargs=2) then
                rng := args[2];
            else
                rng := 0..1;
            end if;
            a,b := op(rng);
            return x->((x-a)/(b-a));
        else
            error "no such color parametrization";
        end if;
    end proc;

    colortab := proc(tab)
        il := sort(map(op,[indices(tab)]));
        if(min(il)<>0 or max(il)<>1000) then
            error "must have endpoints of 0,1000";
        end if;
        l := nops(il);
        ans := table();
        ans[0] := Color(tab[0]);
        for j from 2 to l do
            i1,i2 := il[j-1],il[j];
            col0,col1 := Color(tab[i1]),Color(tab[i2]);
            for i from i1+1 to i2 do
                t := evalf((i-i1)/(i2-i1));
                ans[i] := Color((1-t)*col0[1..3]+t*col1[1..3]);
            end do;
        end do;
        return eval(ans);
    end proc;

    colorvec := proc(tab,bg0,bg1)
        if(not type(args[1],'table')) then
            return colorvec(colorvec0(args[1]),args[2..nargs]);
        end if;
        il := sort(map(op,[indices(tab)]));
        l := nops(il);
        if(l=0) then
            error;
        end if;
        if(nargs=1) then
            return colorvec(tab,tab[il[1]]);
        elif(nargs=2) then
            return colorvec(tab,bg0,tab[il[l]]);
        end if;
        V := Vector(1000);
        for i from 1 to il[1]-1 do
            V[i] := Color(bg0);
        end do;
        for j from 1 to l-1 do
            i1,i2 := il[j],il[j+1];
            col0,col1 := Color(tab[i1]),Color(tab[i2]);
            for i from i1 to i2 do
                t := evalf((i-i1)/(i2-i1));
                V[i] := Color((1-t)*col0[1..3]+t*col1[1..3]);
            end do;
        end do;
        for i from il[l]+1 to 1000 do
            V[i] := Color(bg1);
        end do;
        return V;
    end proc;

    imcmap := proc(im,cmap)
        if(nargs=1) then
            return procname(im,'div1');
        elif(type(cmap,'symbol') or type(cmap,'string')) then
            return imcmap(im,colormap(cmap));
        elif(type(im,'Matrix')) then
            return procname(Array(im,datatype=float[8]),cmap);
        end if;
        m1,m2 := seq(op(2,rng),rng=ArrayTools:-Dimensions(im));
        arr := Array(seq(1..m,m=[m1,m2,3]),datatype=float[8]);
        for i1 from 1 to m1 do
            for i2 from 1 to m2 do
                arr[i1,i2,1],arr[i1,i2,2],arr[i1,i2,3] := cmap(im[i1,i2])[];
            end do;
        end do;
        A := Create(arr);
        if(type(procname,indexed)) then
            fn := op(procname);
            ImageTools:-Write(fn,A);
            return;
        end if;
        return Embed(A);
    end proc;

end module;
