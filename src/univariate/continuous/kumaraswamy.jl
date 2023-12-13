# TODO: should actions be clamped to (0 + _EPISLON, 1 - _EPSILON)?
kumaraswamylogpdf(a::Real, b::Real, x::Real) = kumaraswamylogpdf(promote(a, b, x)...)
function kumaraswamylogpdf(a::T, b::T, x::T) where {T<:Real}
    y = clamp(x, 0, 1)
    val = log(a) + log(b) + xlogy(a - 1, y) + xlog1py(b - 1, -y ^ a)
    return x < 0 || x > 1 ? oftype(val, -Inf) : val
end
