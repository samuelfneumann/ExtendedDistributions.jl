# Adapted from https://github.com/tadamcz/metalogistic/blob/master/metalogistic/main.py
using Printf
using Integrals
using Roots
using LogExpFunctions
using ChainRulesCore

const _METALOG_ε = 1e-24
const _METALOG_PDF_ε = 1e-48

struct Metalogistic{N,T<:AbstractFloat} <: ContinuousUnivariateDistribution
    _coeff::NTuple{N,T}

    function Metalogistic{T}(coeff::T...) where {T}
        N = length(coeff)
        if N < 2
            error("Metalogistic distribution must have at least 2 parameters")
        end
        new{N,T}(NTuple{N}(coeff))
    end
end

function Metalogistic(coeff::T...; check_args::Bool = true) where {T<:Real}
    @Distributions.check_args Metalogistic true
    return Metalogistic{T}(coeff...)
end

function Metalogistic(coeff::Vector{T}; check_args::Bool = true) where {T<:Real}
    return Metalogistic(coeff...; check_args = check_args)
end

function Metalogistic(coeff::Tuple{T}; check_args::Bool = true) where {T<:Real}
    return Metalogistic(coeff...; check_args = check_args)
end

function Metalogistic(coeff::Real...; check_args::Bool=true)
    Metalogistic(promote(coeff...)...; check_args = check_args)
end

function Metalogistic(coeff::Integer...; check_args::Bool=true)
    Metalogistic(float.(coeff)...; check_args=check_args)
end

Distributions.@distr_support Metalogistic -Inf Inf

Distributions.insupport(::Metalogistic, x::Real) = true

#### Conversions
function Base.convert(::Type{Metalogistic{N,T}}, coeff::Real...) where {N,T<:Real}
    Metalogistic(T.(coeff))
end

function Base.convert(::Type{Metalogistic{N,T}}, d::Metalogistic) where {N,T<:Real}
    Metalogistic{T}(T(d._lowerbound), T(d._upperbound), T.(d._coeff))
end

Base.convert(::Type{Metalogistic{N,T}}, d::Metalogistic{N,T}) where {N,T} = d

#### Parameters

Distributions.params(d::Metalogistic) = Tuple(d._coeff)
@inline partype(d::Metalogistic{N,T}) where {N,T<:Real} = T

#### Statistics

function _mean(coeff::T...) where {T<:Real}
    k = length(coeff)
    if k <= 16
        a1 = coeff[1]
        a3 = length(coeff) >=3 ? coeff[3] : 0
        a5 = length(coeff) >=5 ? coeff[5] : 0
        a8 = length(coeff) >=8 ? coeff[8] : 0
        a9 = length(coeff) >=9 ? coeff[9] : 0
        a12 = length(coeff) >=12 ? coeff[12] : 0
        a13 = length(coeff) >=13 ? coeff[13] : 0
        a16 = length(coeff) >=16 ? coeff[16] : 0

        return a1 + (
            10080a3 + 1680a5 + 1680a8 + 252a9 + 322a12 + 45a13 + 66a16
        ) / 20160
    else
        if k % 2 == 1
            expon = (k + 1) // 2
            shift = (2 / (k + 1)) * ((0.5)^expon - (-0.5)^expon)
        else
            integral_bounds = (_METALOG_ε, prevfloat(one(T)))
            p = IntegralProblem(
                (y, p) -> (y - 1//2)^((k // 2) - 1) * logit(y),
                integral_bounds[1],
                integral_bounds[2],
            )
            shift = solve(p, QuadGKJL()).u
        end
        shift = coeff[end] * shift
        μ_km1 = _mean(coeff[begin:end-1]...)
        μ = μ_km1 + shift
        return μ
    end
end

function Distributions.mean(d::Metalogistic{N,T}) where {N,T}
    _mean(params(d)...)
end

Distributions.median(d::Metalogistic) = d._coeff[1]

# #### Sampling
struct MetalogisticSampler{
    S<:Sampleable{Univariate,Continuous}
} <: Sampleable{Univariate,Continuous}
    _dist::S
end

function Distributions.sampler(d::Metalogistic{T}) where T
    return MetalogisticSampler(coeff, Metalogistic(coeff))
end

function Base.rand(rng::AbstractRNG, s::MetalogisticSampler)
    return rand(rng, s._dist)
end

function Base.rand(rng::AbstractRNG, d::Metalogistic)
    u = Base.rand(rng)
    return quantile(d, u)
end

# #### PDFs and CDFs
# p-PDF function
function _metalog_ppdf(y::Real, coeff::T...) where {T}
    if y == one(y) || y == zero(y)
        return zero(y)
    elseif ! (zero(y) < y < one(y))
        error("cumulative probability y must be in (0, 1) but got y = $y")
    end

    k = length(coeff)
    pdf = if k <= 4
        a2 = coeff[2]
        a3 = length(coeff) >= 3 ? coeff[3] : zero(T)
        a4 = length(coeff) >= 4 ? coeff[4] : zero(T)

        one(T) / (
            a2 / (y * (1 - y)) +
            a3 * ((y - 1//2) / (y * (1 - y)) + logit(y)) +
            a4
        )
    else
        if k % 2 == 1
            shift = ((k - 1) / 2) * (y - 1//2)^((k - 3) // 2)
        else
            shift = (
                ((y - 1//2)^((k//2) - 1) / (y * (1 - y))) +
                ((k//2 - 1) * (y - 1//2)^((k//2) - 2) * logit(y))
            )
        end
        p_km1 = _metalog_ppdf(y, coeff[begin:end-1]...)
        shift = coeff[end] * shift
        1 / ((1 / p_km1) + shift)
    end

    # Adjust pdf for numerical errors
    if pdf <= 0
        return pdf + _METALOG_PDF_ε
    end
    return pdf
end

function Distributions.pdf(d::Metalogistic{N,T}, x::Real) where {N,T}
    if x isa Integer
        x = convert(T, x)
    end

    # if x < quantile(d, _METALOG_ε) || x > quantile(d, prevfloat(one(x)))
    #     # TODO: might cause problems with differentiation
    #     return 0.0
    # end

    p = _get_cumulative_prob(d, x)
    p = ChainRulesCore.ignore_derivatives(p)
    return _metalog_ppdf(p, params(d)...)
end

function Distributions.logpdf(d::Metalogistic{N,T}, x::Real) where{N,T}
    if x isa Integer
        x = convert(T, x)
    end

    p = _get_cumulative_prob(d, x)
    p = ChainRulesCore.ignore_derivatives(p)
    return log(_metalog_ppdf(p, params(d)...) + _METALOG_ε)
end

function Distributions.cdf(d::Metalogistic, x::Real)
    cdf = _get_cumulative_prob(d, x)

    # Adjust cdf for numerical errors
    if cdf < 0
        return cdf + eps(cdf)
    end
    return cdf
end

"""
    _get_cumulative_prob(d::Metalogistic, x::Real)

The metalogistic distribution is defined in terms of its quantile function. In order to get
probabilities for a given x-value, like in a traditional CDF, we invert this quantile
function using a numerical equation solver.
"""
function _get_cumulative_prob(d::Metalogistic, x::Real)
    # Adapted from:
    # https://github.com/tadamcz/metalogistic/blob/master/metalogistic/main.py#L576
    f_to_zero = p -> quantile(d, p) - x

    # We need a smaller `xtol` than the default value, in order to ensure correctness when
    # evaluating the CDF or PDF in the extreme tails.
    return ChainRulesCore.ignore_derivatives(
        find_zero(f_to_zero, (zero(x), one(x)), Roots.Brent(); xatol=1e-24),
    )
end

function _metalog_quantile(p::Real, coeffs::T...) where {T}
    if p <= 0
        return -Inf
    elseif p >= 1
        return Inf
    end

    k = length(coeffs)
    # @assert k > 1 "expected k > 1 but got k == $k, $coeffs"
    out = if k == 2
        a1, a2 = coeffs
        a1 + a2 * logit(p)
    elseif k == 3
        a1, a2, a3 = coeffs
        a1 + a2 * logit(p) + a3 * (p - 1//2) * logit(p)
    elseif k == 4
        a1, a2, a3, a4 = coeffs
        (
            a1 +
            a2 * logit(p) +
            a3 * (p - 1//2) * logit(p) +
            a4 * (p - 1//2)
        )
    elseif k % 2 == 1
        (
            _metalog_quantile(p, coeffs[begin:end-1]...) +
            coeffs[end] * (p - 1//2)^((k - 1) // 2)
        )
    else
        (
            _metalog_quantile(p, coeffs[begin:end-1]...) +
            coeffs[end] * (p - 1//2)^((k // 2) - 1) * logit(p)
        )
    end

    return out
end

function Distributions.quantile(d::Metalogistic, p::Real)
    coeffs = Distributions.params(d)
    return _metalog_quantile(p, d._coeff...)
end

"""
    feasible2()
    feasible2(d::Metalogistic{2,T}) where {T}
    feasible2(d::LogitMetalogistic{2,T}) where {T}

Return `ξ` where `a2 >= ξ` to satisfy the feasibility constraint for a metalog
distribution with 2 parameters.
"""
feasible2(a2) = zero(a2)

"""
    feasible3(a2)
    feasible3(d::Metalogistic{3,T}) where {T}
    feasible3(d::LogitMetalogistic{3,T}) where {T}

Return `ξ` where `|a3|/a2 <= ξ` to satisfy the feasibility constraint for a metalog
distribution with 2 parameters.
"""
feasible3(a2) = @assert a2 > 0 "expected a2 > 0 but got a2 = $a2" && 1.66711

"""
    feasible4(a2, a4)
    feasible4(d::Metalogistic{4,T}) where {T}
    feasible4(d::LogitMetalogistic{4,T}) where {T}

Return `ξ` where `|a3|/a2 <= ξ` to satisfy the feasibility constraint for a metalog
distribution with 4 parameters.
"""
function feasible4(a2, a4)
    b = 4.5
    c = 8.5
    d = 1.93

    z = a4/a2
    return if z < -4.0
        error("expected a4/a2 >= -4 but got a4/a2 = $z")
    elseif -4.0 <= z <= 4.5
        d/c * sqrt(c^2 - (z - b)^2)
    elseif z <= 7.0
        0.0216 * (z - 4.5) + 1.93
    elseif z <= 10.0
        0.004 * (z - 7.0) + 1.984
    elseif z <= 30.0
        0.0002 * (z - 10.0) + 1.996
    else
        2.0
    end
end

function feasible4(d::Metalogistic{4,T}) where {T}
    p = params(d)
    feasible4(p[2], p[4])
end

feasible3(d::Metalogistic{3,T}) where {T} = feasible3(params(d)[2])
feasible2(d::Metalogistic{2,T}) where {T} = feasible2()

function isfeasible(d::Metalogistic)::Bool
    k = length(params(d))
    if k > 4
        error(
            "cannot check feasibility for Metalogistic distribution with " *
            "more than 4 parameters"
        )
    elseif k == 2
        return params(d)[2] <= feasible2(d)
    elseif k == 3
        return abs(params(d)[3])/params(d)[2] < feasible3(d)
    else
        return abs(params(d)[3])/params(d)[2] < feasible4(d)
    end
end
