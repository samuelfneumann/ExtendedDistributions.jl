const _GAUSS_OFFSET = 1f-6

"""
    ArctanhNormal(a, b)

The *Hyperbolic Arctangent Normal distribution* is the distribution of a random variable
whose hyperbolic arctangent has a [`Normal`](@ref) distribution.  Or inversely, when
applying the hyperbolic tangent function to a Normal random variable then the resulting
random variable follows a Hyperbolic Arctangent Normal distribution. This distribution is
sometimes called the "squashed Gaussian".  The Hyperbolic Arctangent Normal distribution has
two parameters, the location parameter `μ ∈ ℝ` and the scale parameter `σ ∈ 'ℝ⁺`.
'
```math
f_y(y; a, b) = f_x(g(y)) - (1 - tanh^2(g(y)))
where
```math
g(y) = arctanh(y)
g(y) = x
x ~ Normal(μ, σ)
```

```julia
ArctanhNormal()       # equivalent to ArctanhNormal(0, 1)
ArctanhNormal(μ)      # equivalent to ArctanhNormal(μ, 1)
ArctanhNormal(μ, σ)   # ArctanhNormal distribution with location μ and scale σ

params(d)           # Get the parameters, i.e. (μ, σ)
```

The distribution has bounded support in `(-1, 1)`.

If the location parameter is in `(-1, 1)`, then the mean of the distribution is
approximately equal to the location parameter. If the location parameter equals 0, then the
mean of the distribution is equal to the location parameter. As the location parameter tends
away from 0, the mean of the distribution smoothly deviates from this parameter. If the
location parameter is greater than `1`, then the mean of the distribution is approximately
equal to `1`. If the location parameter is less than `-1`, then the mean of the distribution
is approximately equal to `-1`.

External links

* [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
* [Soft Actor-Critic](https://arxiv.org/abs/1812.05905)
"""
struct ArctanhNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ArctanhNormal{T}(μ::T, σ::T) where {T} = new{T}(μ, σ)
end

function ArctanhNormal(μ::T, σ::T; check_args::Bool=true) where {T<:Real}
    @Distributions.check_args ArctanhNormal (σ, σ > zero(σ))
    return ArctanhNormal{T}(μ, σ)
end

function ArctanhNormal(μ::Real, σ::Real; check_args::Bool=true)
    ArctanhNormal(promote(μ, σ)...; check_args=check_args)
end

function ArctanhNormal(μ::Integer, σ::Integer; check_args::Bool=true)
    ArctanhNormal(float(μ), float(σ); check_args=check_args)
end

function ArctanhNormal(μ::Real; check_args::Bool=true)
    error("not implemented")
    Distributions.@check_args ArctanhNormal (μ, μ > zero(μ))
    ArctanhNormal(μ, one(μ); check_args=false)
end

ArctanhNormal() = ArctanhNormal{Float64}(0.0, 1.0)
Distributions.minimum(::ArctanhNormal{T}) where {T} = -one(T)
Distributions.maximum(::ArctanhNormal{T}) where {T} = one(T)
Distributions.insupport(a::ArctanhNormal, x::Real) = return minimum(a) < x < maximum(a)

#### Conversions
function Base.convert(::Type{ArctanhNormal{T}}, μ::Real, σ::Real) where T<:Real
    ArctanhNormal(T(μ), T(σ))
end

function Base.convert(::Type{ArctanhNormal{T}}, d::ArctanhNormal) where {T<:Real}
    ArctanhNormal{T}(T(d.μ), T(d.σ))
end

Base.convert(::Type{ArctanhNormal{T}}, d::ArctanhNormal{T}) where {T<:Real} = d

#### Parameters

Distributions.params(d::ArctanhNormal) = (d.μ, d.σ)
@inline partype(d::ArctanhNormal{T}) where {T<:Real} = T


#### Statistics

Distributions.mean(d::ArctanhNormal) = ((μ, _) = params(d); tanh(μ))
Distributions.median(d::ArctanhNormal{T}) where {T} = quantile(d, oftype(T, 1//2))

function Distributions.entropy(d::ArctanhNormal{T}) where {T}
    # See  https://github.com/deepmind/rlax/blob/b7d1a012f888d1744245732a2bcf15f38bb7511e/
    #              rlax/_src/distributions.py#L319
    _, σ = params(d)
    _two = one(T) + one(T)

    return log(σ) + oftype(σ, 1//2) * (one(T) + log(_two * π))
end

function atanhnormkldivergence(μ1::Real, σ1::Real, μ2::Real, σ2::Real)
    return atanhnormkldivergence(promote(μ1, σ1, μ2, σ2)...)
end

function atanhnormkldivergence(μ1::T, σ1::T, μ2::T, σ2::T) where {T<:Real}
    # The KL divergence between two ArctanhNormal distributions is equal to the KL
    # divergence between their Normal counterparts. That is, if X and Y follow ArctanhNormal
    # distributions with parameters (μx, σX) and (μy, σy) respectively, then
    #
    #   KL(X || Y) = KL(arctanh(X) || arctanh(Y)) = KL(Normal(μx, σx) || Normal(μy, σy))
    #
    # See  https://github.com/deepmind/rlax/blob/b7d1a012f888d1744245732a2bcf15f38bb7511e/
    #              rlax/_src/distributions.py#L319
    lower, upper = _EPSILON, inv(_EPSILON)
    v1 = clamp(σ1^2, lower, upper)
    v2 = clamp(σ2^2, lower, upper)
    μdiff = μ2 - μ1

    kl_mean = 0.5f0 * μdiff^2 / v2
    kl_cov = 0.5f0 * ((v1/v2) - one(v1) + log(v2) - log(v1))

    return kl_mean + kl_cov
end

function Distributions.kldivergence(p::ArctanhNormal, q::ArctanhNormal)
    μ1, σ1 = params(p)
    μ2, σ2 = params(q)
    return atanhnormkldivergence(μ1, σ1, μ2, σ2)
end

# #### Sampling

struct ArctanhNormalSampler{
        T<:Real,
        S<:Sampleable{Univariate,Continuous},
} <: Sampleable{Univariate,Continuous}
    μ::T
    σ::T
    dist::S
end

function Distributions.sampler(d::ArctanhNormal{T}) where T
    μ, σ = params(d)
    return ArctanhNormalSampler(μ, σ, Normal(μ, σ))
end

function Base.rand(rng::AbstractRNG, s::ArctanhNormalSampler{T}) where {T}
    return tanh(rand(rng, s.dist))
end

function Base.rand(rng::AbstractRNG, d::ArctanhNormal{T}) where T
    (μ, σ) = params(d)
    norm = Normal(μ, σ)
    return tanh(rand(rng, norm))
end

# #### PDFs and CDFs
atanhnormlogpdf(μ::Real, σ::Real, x::Real) = atanhnormlogpdf(promote(μ, σ, x)...)
function atanhnormlogpdf(μ::T, σ::T, x::T) where {T<:Real}
    _x = clamp(x, -one(x) + _GAUSS_OFFSET, one(x) - _GAUSS_OFFSET)
    gauss_x = atanh(_x)
    log_density = normlogpdf(μ, σ, gauss_x)

    shift = log1p(-x^2 + _EPSILON)
    return log_density - shift
end

function Distributions.logpdf(
    d::ArctanhNormal{T}, x::Real; # include_boundary=false,
)::Real where {T}
    μ, σ = params(d)
    return atanhnormlogpdf(μ, σ, x)

     # gauss_x = atanh(clamp(x, -one(x) + _GAUSS_OFFSET, one(x) - _GAUSS_OFFSET))
     # log_density = normlogpdf(μ, σ, gauss_x)
     # shift = log1p(-x^2 + _EPSILON)
     # return log_density - shift
end

function Distributions.pdf(
    d::ArctanhNormal{T}, x::Real; # include_boundary = false,
)::Real where {T}
    μ, σ = params(d)

    gauss_x = atanh(clamp(x, -one(x) + _GAUSS_OFFSET, one(x) - _GAUSS_OFFSET))
    density = normpdf(μ, σ, gauss_x)

    scale = one(x) - x^2 + _EPSILON # ∈ (_EPSILON, 1 + _EPSILON)
    return density / scale
end

function Distributions.cdf(d::ArctanhNormal{T}, x::Real) where {T}
    if x <= minimum(d)
        return zero(T)
    elseif x >= maximum(d)
        return one(T)
    end

    # If Y = f(x) is invertible and monotonically increasing, then P_y(y) = P_x(f⁻¹(y))
    μ, σ = params(d)
    norm = Normal(μ, σ)
    gauss_x = _to_gaussian(x)
    return cdf(norm, gauss_x)
end

function Distributions.quantile(d::ArctanhNormal{T}, q::Real) where {T}
    μ, σ = params(d)
    n = Normal(μ, σ)
    return tanh(quantile(n, q))
end
