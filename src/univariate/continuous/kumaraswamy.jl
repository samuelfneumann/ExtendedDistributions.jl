"""
    Kumaraswamy(a, b)

The *Kumaraswamy distribution* has probability density function

```math
f(x; a, b) = abx^{a-1}(1 - x^a)^{b-1}
```

The Kumaraswamy distribution is related to the [`Beta`](@ref) distribution via the
property that if ``X \\sim \\operatorname{Beta}(\\alpha, \\beta)`` with ``\\alpha = 1``,
then ``Y = X^(1/\\alpha) \\sim \\operatorname{Kumaraswamy}(\\alpha, \\beta)``.

```julia
Kumaraswamy()       # equivalent to Kumaraswamy(1, 1)
Kumaraswamy(a)      # equivalent to Kumaraswamy(a, a)
Kumaraswamy(a, b)   # Kumaraswamy distribution with shape parameters a and b

params(d)           # Get the parameters, i.e. (a, b)
```

External links

* [Kumaraswamy distribution on Wikipedia](http://en.wikipedia.org/wiki/Kumaraswamy_distribution)

"""
struct Kumaraswamy{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    Kumaraswamy{T}(a::T, b::T) where {T} = new{T}(a, b)
end

function Kumaraswamy(a::T, b::T; check_args::Bool=true) where {T<:Real}
    Distributions.@check_args Kumaraswamy (a, a > zero(a)) (b, b > zero(b))
    return Kumaraswamy{T}(a, b)
end

function Kumaraswamy(a::Real, b::Real; check_args::Bool=true)
    Kumaraswamy(promote(a, b)...; check_args=check_args)
end

function Kumaraswamy(a::Integer, b::Integer; check_args::Bool=true)
    Kumaraswamy(float(a), float(b); check_args=check_args)
end

function Kumaraswamy(a::Real; check_args::Bool=true)
    Distributions.@check_args Kumaraswamy (a, a > zero(a))
    Kumaraswamy(a, a; check_args=false)
end

Kumaraswamy() = Kumaraswamy{Float64}(1.0, 1.0)

Base.eltype(::Type{Kumaraswamy{T}}) where {T} = T

Distributions.@distr_support Kumaraswamy 0.0 1.0

#### Conversions
function Base.convert(::Type{Kumaraswamy{T}}, a::Real, b::Real) where T<:Real
    Kumaraswamy(T(a), T(b))
end

function Base.convert(::Type{Kumaraswamy{T}}, d::Kumaraswamy) where {T<:Real}
    Kumaraswamy{T}(T(d.a), T(d.b))
end

Base.convert(::Type{Kumaraswamy{T}}, d::Kumaraswamy{T}) where {T<:Real} = d

#### Parameters

Distributions.params(d::Kumaraswamy) = (d.a, d.b)
@inline partype(d::Kumaraswamy{T}) where {T<:Real} = T


#### Statistics

function Distributions.mean(d::Kumaraswamy{T}) where {T}
    one_over_a = oneunit(T) / d.a

    numerator = d.b * gamma(oneunit(T) + one_over_a) * gamma(d.b)
    denominator = gamma(oneunit(T) + one_over_a + d.b)

    return numerator / denominator
end

function Distributions.mode(d::Kumaraswamy{T}; check_args::Bool=true) where {T}
    a, b = params(d)
    aequalbequal1 = a == b == oneunit(T)
    Distributions.@check_args(
        Kumaraswamy,
        (a, a >= 1, "mode is defined only when a >= 1."),
        (b, b >= 1, "mode is defined only when b >= 1."),
        ((a, b), (a, b) == (oneunit(T), oneunit(T)), "mode is defined only when b >= 1."),
    )

    return ((a - oneunit(T)) / (a * b - oneunit(T)))^inv(a)
end

Distributions.modes(d::Kumaraswamy) = [mode(d)]

function Distributions.var(d::Kumaraswamy{T}) where {T}
    # See https://en.wikipedia.org/wiki/Kumaraswamy_distribution
    (a, b) = params(d)
    _one = oneunit(T)
    _two = _one + _one

    m1 = _rawmoment(d, 1)
    m2 = _rawmoment(d, 2)

    return m2 - m1^2
end

function Distributions.entropy(d::Kumaraswamy{T}) where {T}
    a, b = params(d)

    _one = oneunit(T)
    h_b = sum(i -> _one/i, _one:b)

    return (_one - _one/a) + (_one - _one/b) * h_b - log(a * b)
end

function Distributions.skewness(d::Kumaraswamy{T}) where {T}
    (a, b) = params(d)

    if a == b
        return zero(a)
    else
        m3 = _centralmoment(d, 3)
        σ = sqrt(var(d))
        return m3 / σ^3
    end
end

function Distributions.quantile(d::Kumaraswamy{T}, q::Real) where {T}
    a, b = params(d)
    if !(zero(q) <= q <= oneunit(q))
        throw(DomainError(q, "expected q ∈ [0, 1] but got $q"))
    end

    _one = oneunit(T)
    return (_one - (_one - q)^inv(b))^inv(a)
end

function Distributions.cdf(d::Kumaraswamy{T}, x::Real) where {T}
    if x <= minimum(d)
        return zero(T)
    elseif x >= maximum(d)
        return one(T)
    end
    a, b = params(d)
    _one = oneunit(T)

    return _one - (_one - x^a)^b
end

function Distributions.pdf(d::Kumaraswamy{T}, x::Real) where {T}
    if !insupport(d, x)
        return zero(T)
    end

    a, b = params(d)
    _one = oneunit(T)

    return a * b * x^(a-_one) * (_one - x^a)^(b-_one)
end

function Distributions.logpdf(d::Kumaraswamy{T}, x::Real) where {T}
    if !insupport(d, x)
        return -Inf
    end

    a, b = params(d)
    _one = oneunit(T)

    return log(a) + log(b) + (a - _one) * log(x) + (b - _one) * log1p(-x^a)
end

function Distributions.median(d::Kumaraswamy{T}) where {T}
    a, b = params(d)
    _one = oneunit(T)
    _two = _one + _one

    return (_one - _two^-inv(b))^inv(a)
end

Distributions.minimum(::Kumaraswamy{T}) where {T} = zero(T)
Distributions.maximum(::Kumaraswamy{T}) where {T} = oneunit(T)
Distributions.insupport(::Kumaraswamy, x::Real) = zero(x) < x < oneunit(x)

function _centralmoment(d::Kumaraswamy{T}, n) where {T}
    if n <= 0
        error("cannot take 0-th moment")
    elseif n == 1
        mean(d)
    elseif n == 2
        var(d)
    elseif n == 3
        m3 = _rawmoment(d, 3)
        m2 = _rawmoment(d, 2)
        m1 = _rawmoment(d, 1)
        return m3 - 3 * m1 * m2 + 2 * m1^3
    elseif n == 4
        m4 = _rawmoment(d, 4)
        m3 = _rawmoment(d, 3)
        m2 = _rawmoment(d, 2)
        m1 = _rawmoment(d, 1)
        return m4 - 4 * m1 * m3 + 6 * m1^2 * m2 - 3 * m1^4
    elseif n > 4
        _minusone = -oneunit(T)
        return sum(
            j -> binomial(Int(n), int(j)) * (_minusone)^(n-j) * _rawmoment(d, j) * _centralmoment(n-j),
            zero(T):convert(T, n),
        )
    end
end

function _rawmoment(d::Kumaraswamy{T}, n) where {T}
    _one = oneunit(T)
    mn = b * beta(_one + n/a, b)
end

#### Sampling

struct KumaraswamySampler{T<:Real} <: Sampleable{Univariate,Continuous}
    ia::T
    ib::T
end

function Distributions.sampler(d::Kumaraswamy{T}) where T
    (a, b) = params(d)
    return KumaraswamySampler{T}(inv(a), inv(b))
end

function Base.rand(rng::AbstractRNG, s::KumaraswamySampler{T}) where {T}
    ia = s.ia
    ib = s.ib

    y = rand(rng) # ~ U(0, 1)
    _one = oneunit(T)

    out = (_one - (_one - y)^ib)^ia
    if T === Float32
        out = convert(T, out)
    end
    return out
end

function Base.rand(rng::AbstractRNG, d::Kumaraswamy{T}) where T
    if T === Float32
        y = rand(T, rng) # ~ U(0, 1)
    else
        y = rand(rng)
    end
    return quantile(d, y)
end
