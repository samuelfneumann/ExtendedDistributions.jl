const discrete_distributions = []

const continuous_distributions = [
    "arctanhnormal",
    "metalogistic",
    "logitmetalogistic",
    # "kumaraswamy",
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

@eltype Arcsine
@eltype ArctanhNormal
@eltype Bernoulli
@eltype BernoulliLogit
@eltype Beta
@eltype BetaBinomial
@eltype BetaPrime
@eltype Binomial
@eltype Biweight
@eltype Categorical
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
