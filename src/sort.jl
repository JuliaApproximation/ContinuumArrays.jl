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


