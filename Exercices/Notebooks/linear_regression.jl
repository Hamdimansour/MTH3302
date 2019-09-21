using Statistics, Gadfly, LinearAlgebra, Distributions
function estimate(d::Array{Float64}, Y::Array{Float64,1})
    n = size(d)[1]::Int
    X = [ones(n) d]
    β = (X'X)\X'Y
    
    return β
end
function predict(d, beta)
    n = size(d)[1]::Int
    X = [ones(n) d]
    y = X*beta
    return y
end
function R_2(X, y, beta)
    n = size(X)[1]::Int
    ŷ = predict(X, beta)
    ȳ = mean(y)
    p = size(beta)[1]-1
    SSᵣ = (ŷ .- ȳ)'*(ŷ .- ȳ)
    SSₜ = (y .- ȳ)'*(y .- ȳ)
    R² = SSᵣ / SSₜ
    return R²
end
function R_ajuste(X, y, beta)
    n = size(X)[1]::Int
    ŷ = predict(X, beta)
    ȳ = mean(y)
    p = size(beta)[1]-1
    SSₑ = (y - ŷ)'*(y - ŷ)
    SSₜ = (y .- ȳ)'*(y .- ȳ)
    R²ajusté = 1 - (SSₑ/(n-p))/(SSₜ/(n-1))
    return R²ajusté
end
function R_prev(d, y, beta)
    n = size(d)[1]::Int
    X = [ones(n) d]
    H = X/(X'X)*X'
    ŷ = predict(d, beta)
    e = y - ŷ 
    ẽ = e./(1 .- diag(H))
    SSₜ = (y .- ȳ)'*(y .- ȳ)
    R²prev = 1 - (ẽ'ẽ)/SSₜ
end
function graph_residus(X, y, beta)
    ŷ = predict(X, beta)
    e = y - ŷ
    plot(x=ŷ, y=e)
end
function intervalle_confiance(X, y, beta, alpha)
    ŷ = predict(X, beta)
    ȳ = mean(y)
    p = size(beta)[1]-1
    n = size(X)[1]::Int
    σ² = (y - ŷ)'*(y - ŷ)/(n-p-1)
    
    X = hcat(ones(length(data.Final)), data.CP1, data.CP2, session)
    c = diag(inv(X'X))
    t = cdf(TDist(n-p-1), alpha)
    
    Δβ = t.*sqrt.(σ².*c)
    return [(β - Δβ) (β + Δβ)][2:end,:]
end
function test_fisher(X::Array{Float64}, y::Array{Float64,1}, beta::Array{Float64,1}, alpha::Float64)
    ŷ = predict(X, beta)
    ȳ = mean(y)
    SSₑ = (y - ŷ)'*(y - ŷ)
    SSᵣ = (ŷ .- ȳ)'*(ŷ .- ȳ)
    p = size(beta)[1]-1
    n = size(X)[1]::Int
    F₀ = (SSᵣ/p)/(SSₑ/(n-p-1))
    return ccdf(FDist(p, n-p-1), F₀) < alpha
end
