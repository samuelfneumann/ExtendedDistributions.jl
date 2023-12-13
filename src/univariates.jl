const discrete_distributions = []

const continuous_distributions = [
    "arctanhnormal",
    "metalogistic",
    "logitmetalogistic",
    "kumaraswamy",
    "laplace",
]

for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end

for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end


# Override the default eltype implementation to reflect the true type that is returned when
# sampling from these distributions
macro eltype(type)
   :(Base.eltype(::Type{<:$type{T}}) where {T} = T)
end

macro analytical_kl(type)
   :(analytical_kl(::Type{<:$type}) = true)
end

macro no_analytical_kl(type)
   :(analytical_kl(::Type{<:$type}) = false)
end

macro analytical_entropy(type)
   :(analytical_entropy(::Type{<:$type}) = true)
end

macro no_analytical_entropy(type)
   :(analytical_entropy(::Type{<:$type}) = false)
end

"""
    analytical_kl(D)::Bool
    analytical_kl(::D)::Bool

Returns whether or not there is an analytical form for the KL divergence between two
distributions with type D
"""
function analytical_kl end
analytical_kl(x::UnivariateDistribution) = analytical_kl(typeof(x))

"""
    analytical_entropy(D)::Bool
    analytical_entropy(::D)::Bool

Returns whether or not there is an analytical form for the entropy between two distributions
with type D
"""
analytical_entropy(x::UnivariateDistribution) = analytical_entropy(typeof(x))

@analytical_entropy ArctanhNormal
@analytical_entropy Bernoulli
@analytical_entropy Beta
@analytical_entropy Dirichlet
@analytical_entropy Exponential
@analytical_entropy Gamma
@analytical_entropy Kumaraswamy
@analytical_entropy Laplace
@analytical_entropy Normal

@analytical_kl ArctanhNormal
@analytical_kl Beta
@analytical_kl Dirichlet
@analytical_kl Exponential
@analytical_kl Gamma
@analytical_kl Laplace
@analytical_kl Normal

@no_analytical_kl Kumaraswamy

@eltype Arcsine
@eltype ArctanhNormal
@eltype Bernoulli
@eltype BernoulliLogit
@eltype Beta
@eltype BetaBinomial
@eltype BetaPrime
@eltype Binomial
@eltype Biweight
@eltype Cauchy
@eltype Chi
@eltype Chisq
@eltype Cosine
@eltype DoubleExponential
@eltype Erlang
@eltype Exponential
@eltype FDist
@eltype Frechet
@eltype Gamma
@eltype GeneralizedPareto
@eltype GeneralizedExtremeValue
@eltype Geometric
@eltype InverseWishart
@eltype InverseGamma
@eltype InverseGaussian
@eltype JohnsonSU
@eltype Kumaraswamy
