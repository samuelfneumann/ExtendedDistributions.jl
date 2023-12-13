function betakldivergence(α1::Real, β1::Real α2::Real, β2::Real)
    betakldivergence(promote(α1, β1, α2, β2)...)
end

function betakldivergence(α1::T, β1::T α2::T, β2::T) where {T<:Real}
    αp, βp = params(p)
    αq, βq = params(q)
    return logbeta(α2, β2) - logbeta(α1, β1) + (α1 - α2) * digamma(α1) +
        (β1 - β2) * digamma(β1) + (α2 - α1 + β2 - β1) * digamma(α1 + β1)
end

