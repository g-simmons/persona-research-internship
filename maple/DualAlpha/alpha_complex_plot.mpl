unprotect('plot'); plot := proc (p::{array, list, rtable, set, algebraic, 
procedure, And(`module`,appliable)} := [], {axiscoordinates::anything := NULL,
coords::anything := NULL, ispolarplot::boolean := false}) local cname; if 
coords <> NULL then cname := convert(coords,'string'); if not ispolarplot and 5
<= length(cname) and cname[1 .. 5] = "polar" and axiscoordinates = 'polar' then
return plots:-polarplot(p,_rest) end if end if; try return Plot:-Plot2D(Plot:-
Preprocess(procname,p,_rest,`if`(coords <> NULL,_options['coords'],NULL),`if`(
axiscoordinates <> NULL,_options['axiscoordinates'],NULL))) catch: error end 
try end proc; setattribute('plot',protected, _syslib);
