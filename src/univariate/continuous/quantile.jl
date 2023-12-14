# ################################################################### 
# Kumaraswamy
# ################################################################### 
@dist_args kumaraswamyquantile Kumaraswamy
@promote kumaraswamyquantile
function kumaraswamyquantile(a::T, b::T, q::T) where {T<:Real}
    return (1 - (1 - q)^inv(b))^inv(a)
end

# ################################################################### 
# LogitNormal
# ################################################################### 
@dist_args logitnormquantile LogitNormal
@promote logitnormquantile
function logitnormquantile(μ::T, σ::T, q::T) where {T<:Real}
    return logistic(norminvcdf(μ, σ, q))
end

# ################################################################### 
# Laplace
# ################################################################### 
@dist_args laplacequantile Laplace
@promote laplacequantile
function laplacequantile(μ::Real, θ::Real, q::Real)
    q < 1/2 ? laplace_xval(μ, θ, log(2q)) : laplace_xval(μ, θ, -log(2(1 - q)))
end

# ################################################################### 
# ArctanhNormal
# ################################################################### 
@dist_args atanhnormquantile ArctanhNormal
@promote atanhnormquantile
function atanhnormquantile(μ::T, σ::T, q::T) where {T<:Real}
    return tanh(norminvcdf(μ, σ, q))
end

# ################################################################### 
# Normal
# ################################################################### 
@dist_args normquantile ArctanhNormal
@promote normquantile
normquantile(μ::T, σ::T, q::T) where {T} = norminvcdf(μ, σ, q)

# ################################################################### 
# Logistic
# ################################################################### 
@dist_args logisticquantile Logistic
@promote logisticquantile
function logisticquantile(μ::T, θ::T, q::T) where {T<:Real}
    return logistic_xval(μ, θ, logit(q))
end
