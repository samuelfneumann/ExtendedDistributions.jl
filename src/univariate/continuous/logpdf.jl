# ###################################################################
# LogUniform
# ###################################################################
@dist_args loguniformlogpdf LogUniform
@promote loguniformlogpdf
function loguniformlogpdf(a::T, b::T, x::T)::T where {T<:Real}
loguniformlogpdf
    if a <= x <= b
        return -log(x * log(b/a))
    else
        return -T(Inf)
    end
end

# ###################################################################
# LogitNormal
# ###################################################################
@dist_args logitnormlogpdf LogitNormal
@promote logitnormlogpdf
function logitnormlogpdf(μ::T, σ::T, x::T)::T where {T<:Real}
    if zero(x) < x < one(x)
        lx = logit(x)
        return normlogpdf(μ, σ, lx) - log(x) - log1p(-x)
    else
        return -T(Inf)
    end
end

# ###################################################################
# Logistic
# ###################################################################
logistic_zval(μ, θ, x) = (x - μ) / θ
logistic_xval(μ, θ, z) = μ + z * θ

@dist_args logisticlogpdf Logistic
@promote logisticlogpdf
function logisticlogpdf(μ::T, θ::T, x::T)::T where {T<:Real}
    return (u = -abs(logistic_zval(μ, θ, x)); u - 2*log1pexp(u) - log(θ))
end

# ###################################################################
# Laplace
# ###################################################################
laplace_zval(μ, θ, x) = (x - μ) / θ
laplace_xval(μ, θ, z) = μ + z * θ

@dist_args laplacelogpdf Laplace
@promote laplacelogpdf
function laplacelogpdf(μ::T, θ::T, x::T)::T where {T<:Real}
    return -(abs(laplace_zval(μ, θ, x)) + log(2*θ))
end

# ###################################################################
# ArctanhNormal
# ###################################################################
@dist_args atanhnormlogpdf ArctanhNormal
@promote atanhnormlogpdf
function atanhnormlogpdf(μ::T, σ::T, x::T)::T where {T<:Real}
    _x = clamp(x, -one(x) + _GAUSS_OFFSET, one(x) - _GAUSS_OFFSET)
    gauss_x = atanh(_x)
    log_density = normlogpdf(μ, σ, gauss_x)

    shift = log1p(-x^2 + _EPSILON)
    return log_density - shift
end

# ###################################################################
# Kumaraswamy
# ###################################################################
@dist_args kumaraswamylogpdf Kumaraswamy
@promote kumaraswamylogpdf
function kumaraswamylogpdf(a::T, b::T, x::T)::T where {T<:Real}
    y = clamp(x, 0, 1)
    val = log(a) + log(b) + xlogy(a - 1, y) + xlog1py(b - 1, -y ^ a)
    return x < 0 || x > 1 ? oftype(val, -Inf) : val
end
