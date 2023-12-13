function normkldivergence(μ1::Real, σ1::Real, μ2::Real, σ2::Real)
    return atanhnormkldivergence(promote(μ1, σ1, μ2, σ2)...)
end

function normkldivergence(μ1::T, σ1::T, μ2::T, σ2::T) where {T<:Real}
    lower, upper = _EPSILON, inv(_EPSILON)
    v1 = clamp(σ1^2, lower, upper)
    v2 = clamp(σ2^2, lower, upper)
    μdiff = μ2 - μ1

    kl_mean = 0.5f0 * μdiff^2 / v2
    kl_cov = 0.5f0 * ((v1/v2) - one(v1) + log(v2) - log(v1))

    return kl_mean + kl_cov
end
