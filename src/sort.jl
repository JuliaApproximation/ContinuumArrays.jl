# gives a generalization of midpoint for when `a` or `b` is infinite
function genmidpoint(a::T, b::T) where T
    if isinf(a) && isinf(b)
        zero(T)
    elseif isinf(a)
        b - 100
    elseif isinf(b)
        a + 100
    else
        (a+b)/2
    end
end


function searchsortedfirst_layout(::ExpansionLayout, f, x; iterations=47)
    d = axes(f,1)
    a,b = first(d), last(d)

    for k=1:iterations  #TODO: decide 47
        m= genmidpoint(a,b)
        (f[m] ≤ x) ? (a = m) : (b = m)
    end
    (a+b)/2
end

# other quasi-vectors, e.g. broadcasted functions, are expanded in their natural basis
for find in (:findall, :findfirst, :findlast)
    find_layout = Symbol(find, "_layout")
    @eval $find_layout(::Any, f, v; kwds...) = $find(f, expand(v); kwds...)
end

findall_layout(::ExpansionLayout, f, v; kwds...) = error("Overload findall_layout(::$(typeof(MemoryLayout(v))), ::$(typeof(f)), v)")
findfirst_layout(::ExpansionLayout, f, v; kwds...) = _first_or_nothing(findall(f, v; kwds...))
findlast_layout(::ExpansionLayout, f, v; kwds...) = _last_or_nothing(findall(f, v; kwds...))
_first_or_nothing(r) = isempty(r) ? nothing : first(r)
_last_or_nothing(r) = isempty(r) ? nothing : last(r)
