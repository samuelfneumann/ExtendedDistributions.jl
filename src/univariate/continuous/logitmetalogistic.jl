struct LogitMetalogistic{N,T<:AbstractFloat} <: ContinuousUnivariateDistribution
    _coeff::NTuple{N,T}
    _lowerbound::T
    _upperbound::T

    function LogitMetalogistic{T}(lb::T, ub::T, coeff::T...) where {T}
        N = length(coeff)
        if N < 2
            error("LogitMetalogistic distribution must have at least 2 parameters")
        end

        @assert lb < ub "cannot have lower bound greater than or equal to upper bound"

        new{N,T}(NTuple{N}(coeff), lb, ub)
    end
end

function LogitMetalogistic(lb::T, ub::T, coeff::T...; check_args::Bool = true) where {T<:Real}
    @Distributions.check_args LogitMetalogistic true
    return LogitMetalogistic{T}(lb, ub, coeff...)
end

function LogitMetalogistic(
    lb::T, ub::T, coeff::Vector{T}; check_args::Bool = true,
) where {T<:Real}
    return LogitMetalogistic(lb, ub, coeff...; check_args = check_args)
end

function LogitMetalogistic(
    lb::T, ub::T, coeff::Tuple{T}; check_args::Bool = true,
) where {T<:Real}
    return LogitMetalogistic(lb, ub, coeff...; check_args = check_args)
end

function LogitMetalogistic(lb::Real, ub::Real, coeff::Real...; check_args::Bool=true)
    LogitMetalogistic(promote(lb, ub, coeff...)...; check_args = check_args)
end

function LogitMetalogistic(lb::Real, ub::Real, coeff::Vector{Real}; check_args::Bool=true)
    LogitMetalogistic(promote(lb, ub, coeff...)...; check_args = check_args)
end

function LogitMetalogistic(lb::Real, ub::Real, coeff::Tuple{Real}; check_args::Bool=true)
    LogitMetalogistic(promote(lb, ub, coeff...)...; check_args = check_args)
end

function LogitMetalogistic(args...; kwargs...)
    LogitMetalogistic(0f0, 1f0, args...; kwargs...)
end

Distributions.minimum(d::LogitMetalogistic) = d._lowerbound
Distributions.maximum(d::LogitMetalogistic) = d._upperbound
function Distributions.insupport(d::LogitMetalogistic, x::Real)
    Distributions.minimum(d) <= x <= Distributions.maximum(d)
end

#### Conversions
function Base.convert(
    d::Type{LogitMetalogistic{N,T}}, lb::Real, ub::Real, coeff::Real...,
) where {N,T<:Real}
    LogitMetalogistic(T(lb), T(ub), T.(coeff))
end

function Base.convert(
    ::Type{LogitMetalogistic{N,T}}, d::LogitMetalogistic,
) where {N,T<:Real}
    LogitMetalogistic{T}(
        T(Distributions.minimum(d)), T(Distributions.maximum(d)), T.(d._coeff),
    )
end

Base.convert(::Type{LogitMetalogistic{N,T}}, d::LogitMetalogistic{N,T}) where {N,T} = d

#### Parameters

Distributions.params(d::LogitMetalogistic) = Tuple(d._coeff)
@inline partype(d::LogitMetalogistic{N,T}) where {N,T<:Real} = T

#### Statistics

function Distributions.median(d::LogitMetalogistic)
    lb = Distributions.minimum(d)
    ub = Distributions.maximum(d)
    a1 = params(d)[1]

    return (lb + ub * exp(a1)) / (1 + exp(a1))
end

# #### Sampling
struct LogitMetalogisticSampler{
    S<:Sampleable{Univariate,Continuous}
} <: Sampleable{Univariate,Continuous}
    _dist::S
end

function Distributions.sampler(d::LogitMetalogistic{T}) where T
    return LogitMetalogisticSampler(coeff, LogitMetalogistic(coeff))
end

function Base.rand(rng::AbstractRNG, s::LogitMetalogisticSampler)
    return rand(rng, s._dist)
end

function Base.rand(rng::AbstractRNG, d::LogitMetalogistic{N,T}) where {N,T}
    u = Base.rand(rng)
    x = quantile(d, u)

    # Due to numerical instabilities, the sampled value may get rounded to one of the bounds
    return if x == d._lowerbound
        # Or should we add _METALOG_ε?
        x = nextfloat(convert(T, d._lowerbound))
    elseif x == d._upperbound
        # Or should we subtract _METALOG_ε?
        x = prevfloat(convert(T, d._upperbound))
    else
        x
    end
end

# #### PDFs and CDFs
# p-PDF function
function Distributions.pdf(d::LogitMetalogistic{N,T}, x::Real; check=true) where {N,T}
    if x isa Integer
        x = convert(T, x)
    end

    ub = Distributions.maximum(d)
    lb = Distributions.minimum(d)

    if x <= lb || x >= ub
        return zero(x)
    end

    # Get the cumulative probability of x
    y = _get_cumulative_prob(d, x)

    # Bound the cumulative probability to be in (0, 1). The CDF is 1 only at the
    # upper bound and 0 only at the lower bound, but the random variable x ∈ (lower bound,
    # upper bound)
    if y == one(y)
        y = prevfloat(one(y))
    elseif y ==  zero(y)
        y = _METALOG_PDF_ε
    end

    y = ChainRulesCore.ignore_derivatives(y)

    m = _metalog_ppdf(y, params(d)...)
    M = _metalog_quantile(y, d._coeff...)
    if check
        @assert isfinite(M)
        @assert isfinite(m)
    end

    return m * (
        ((1 + exp(M))^2) /
        ((ub - lb) * exp(M))
    )
end

function Distributions.logpdf(d::LogitMetalogistic{N,T}, x::Real; check=true) where {N,T}
    if x isa Integer
        x = convert(T, x)
    end

    ub = Distributions.maximum(d)
    lb = Distributions.minimum(d)

    if x <= lb || x >= ub
        return -Inf
    end

    # Get the cumulative probability of x
    y = _get_cumulative_prob(d, x)

    # Bound the cumulative probability to be in (0, 1). The CDF is 1 only at the
    # upper bound and 0 only at the lower bound, but the random variable x ∈ (lower bound,
    # upper bound)
    if y == one(y)
        y = prevfloat(one(y))
    elseif y ==  zero(y)
        y = _METALOG_PDF_ε
    end

    y = ChainRulesCore.ignore_derivatives(y)

    m = _metalog_ppdf(y, params(d)...)
    M = _metalog_quantile(y, params(d)...)
    if check
        @assert isfinite(M)
        @assert isfinite(m)
    end

    return (log(m) +
        2 * log1p(exp(M)) -
        log((ub - lb)) - M)
end

Distributions.cdf(d::LogitMetalogistic, x::Real) = _get_cumulative_prob(d, x)

"""
    _get_cumulative_prob(d::LogitMetalogistic, x::Real)

The metalogistic distribution is defined in terms of its quantile function. In order to get
probabilities for a given x-value, like in a traditional CDF, we invert this quantile
function using a numerical equation solver.
"""
function _get_cumulative_prob(d::LogitMetalogistic, x::Real)
    if !insupport(d, x)
        error("x not in support [$(Distributions.minimum(d)), $(Distributions.maximum(d))]")
    end

    # Adapted from:
    # https://github.com/tadamcz/metalogistic/blob/master/metalogistic/main.py#L576
    f_to_zero = p -> quantile(d, p) - x

     # We need a smaller `xtol` than the default value, in order to ensure correctness when
     # evaluating the CDF or PDF in the extreme tails.
     return ChainRulesCore.ignore_derivatives(
         find_zero(f_to_zero, (zero(x), one(x)), Roots.Brent(); xatol=1e-24),
     )
end

function _logit_metalog_quantile(p::Real, lb::T, ub::T, coeffs::T...) where {T}
    if p == 0
        return lb
    elseif p < 0
        error("expected y ∈ [0, 1] but got y = $y")
    elseif p == 1
        return ub
    elseif p > 1
        error("expected y ∈ [0, 1] but got y = $y")
    else
        M = _metalog_quantile(p, coeffs...)
        @assert isfinite(M)
        return (lb + ub * exp(M)) / (1 + exp(M))
    end
end

function Distributions.quantile(d::LogitMetalogistic, p::Real)
    return _logit_metalog_quantile(
        p, Distributions.minimum(d), Distributions.maximum(d), Distributions.params(d)...,
    )
end

function feasible4(d::LogitMetalogistic{4,T}) where {T}
    p = params(d)
    feasible4(p[2], p[4])
end

feasible3(d::LogitMetalogistic{3,T}) where {T} = feasible3(params(d)[2])
feasible2(d::LogitMetalogistic{2,T}) where {T} = feasible2(params(d)[2])

function isfeasible(d::LogitMetalogistic)::Bool
    k = length(params(d))
    if k > 4
        error(
            "cannot check feasibility for Metalogistic distribution with " *
            "more than 4 parameters"
        )
    elseif k == 2
        return params(d)[2] > feasible2(d)
    elseif k == 3
        return abs(params(d)[3])/params(d)[2] < feasible3(d)
    else
        return abs(params(d)[3])/params(d)[2] < feasible4(d)
    end
end
