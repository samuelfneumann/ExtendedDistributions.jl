laplace_zval(μ, θ, x) = (x - μ) / θ
laplace_xval(μ, θ, x) = μ + z * θ

laplacelogpdf(μ::Real, θ::Real, x::Real) = laplacelogpdf(promote(μ, θ, x)...)
function laplacelogpdf(μ::T, θ::T, x::T) where {T<:Real}
    return -(abs(zval(μ, θ, x)) + log(2*θ))
end
