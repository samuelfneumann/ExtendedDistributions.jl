# #################################################################### 
# LogitNormal
# #################################################################### 
@dist_args logitnormkldivergence LogitNormal
@promote logitnormkldivergence
function logitnormkldivergence(μ1::T, σ1::T, μ2::T, σ2::T) where {T<:Real}
    return normkldivergence(μ1, σ1, μ2, σ2)
end

# #################################################################### 
# Laplace
# #################################################################### 
@dist_args laplacekldivergence Laplace
@promote laplacekldivergence
function laplacekldivergence(μ1::T, θ1::T, μ2::T, θ2::T) where {T<:Real}
    r = abs(μ1 - μ2)
    return (θ1 * exp(-r / θ1) + r) / θ2 + log(θ2 / θ1) - 1
end


# #################################################################### 
# Normal
# #################################################################### 
@dist_args normkldivergence Normal
@promote normkldivergence
function normkldivergence(μ1::T, σ1::T, μ2::T, σ2::T) where {T<:Real}
    lower, upper = _EPSILON, inv(_EPSILON)
    v1 = clamp(σ1^2, lower, upper)
    v2 = clamp(σ2^2, lower, upper)
    μdiff = μ2 - μ1

    kl_mean = 0.5f0 * μdiff^2 / v2
    kl_cov = 0.5f0 * ((v1/v2) - one(v1) + log(v2) - log(v1))

    return kl_mean + kl_cov
end

# #################################################################### 
# ArctanhNormal
# #################################################################### 
@dist_args atanhnormkldivergence ArctanhNormal
@promote atanhnormkldivergence
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

# #################################################################### 
# Beta
# #################################################################### 
@dist_args betakldivergence Beta
@promote betakldivergence
function betakldivergence(α1::T, β1::T, α2::T, β2::T) where {T<:Real}
    return logbeta(α2, β2) - logbeta(α1, β1) + (α1 - α2) * digamma(α1) +
        (β1 - β2) * digamma(β1) + (α2 - α1 + β2 - β1) * digamma(α1 + β1)
end

