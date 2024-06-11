# ###################################################################
# LogUniform
# ###################################################################
@dist_args loguniformccdf LogUniform
@promote loguniformccdf
function loguniformccdf(a::T, b::T, y::T)::T where {T<:Real}
    1 - loguniformcdf(a, b, y)
end

@dist_args loguniformcdf LogUniform
@promote loguniformcdf
function loguniformcdf(a::T, b::T, y::T)::T where {T<:Real}
    _y = clamp(y, a, b)
    return log(log(_y / a)) - log(log(b / a))
end

# ###################################################################
# Kumaraswamy
# ###################################################################
@dist_args kumaraswamyccdf Kumaraswamy
@promote kumaraswamyccdf
function kumaraswamyccdf(a::T, b::T, y::T)::T where {T<:Real}
    _y = (1 - clamp(y, 0, 1)^a)^b
    return y < 0 ? one(T) : (y > 1 ? zero(T) : _y)
end

@dist_args kumaraswamycdf Kumaraswamy
@promote kumaraswamycdf
function kumaraswamycdf(a::T, b::T, y::T)::T where {T<:Real}
    1 - kumaraswamycdf(a, b, y)
end

# ###################################################################
# Logistic
# ###################################################################
@dist_args logisticccdf Logistic
@promote logisticccdf
function logisticccdf(μ::T, θ::T, y::T)::T where {T<:Real}
    return logistic(-logistic_zval(μ, θ, y))
end

@dist_args logisticdf Logistic
@promote logisticdf
function logisticdf(μ::T, θ::T, y::T)::T where {T<:Real}
    return logistic(logistic_zval(μ, θ, y))
end

# ###################################################################
# LogitNormal
# ###################################################################
@dist_args logitnormccdf LogitNormal
@promote logitnormccdf
function logitnormccdf(μ::T, σ::T, y::T)::T where {T<:Real}
    return y ≤ 0 ? zero(T) : y ≥ 1 ? one(T) : normcdf(μ, σ, logit(y))
end

@dist_args logitnormcdf LogitNormal
@promote logitnormcdf
function logitnormcdf(μ::T, σ::T, y::T)::T where {T<:Real}
    return y ≤ 0 ? one(T) : y ≥ 1 ? zero(T) : normcdf(μ, σ, logit(y))
end

# ###################################################################
# Laplace
# ###################################################################
@dist_args laplaceccdf Laplace
@promote laplaceccdf
function laplaceccdf(μ::Real, θ::Real, y::Real)::T
    (z = laplace_zval(μ, θ, y); z > 0 ? exp(-z)/2 : 1 - exp(z)/2)
end

@dist_args laplacecdf Laplace
@promote laplacecdf
function laplacecdf(μ::Real, θ::Real, y::Real)::T
    (z = laplace_zval(μ, θ, y); z < 0 ? exp(z)/2 : 1 - exp(-z)/2)
end

# ###################################################################
# ArctanhNormal
# ###################################################################
@dist_args atanhnormccdf ArctanhNormal
@promote atanhnormccdf
function atanhnormccdf(μ::T, σ::T, y::T)::T where {T<:Real}
    return y ≤ 0 ? zero(T) : y ≥ 1 ? one(T) : normcdf(μ, σ, atanh(y))
end

@dist_args atanhnormcdf ArctanhNormal
@promote atanhnormcdf
function atanhnormcdf(μ::T, σ::T, y::T)::T where {T<:Real}
    return y ≤ 0 ? one(T) : y ≥ 1 ? zero(T) : normcdf(μ, σ, atanh(y))
end
