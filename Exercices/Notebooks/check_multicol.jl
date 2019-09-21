function check_multicol(X::Array{T,2} where T<:Real)
   
    # Standardize first the explanatory variables
    m = mean(X, dims=1)
    s = std(X, dims=1)
    
    m[1] = 0
    s[1] = 1
    
    X̃ = (X .- m) ./s
    
    
    # compute the singular values
    λ = svdvals(X̃)
    # Calculer les valeurs singulières de X est plus efficace que calculer les valeurs propres de X'X
    
    # Coompute the multicollinearity index
    ϕ = maximum(λ) / minimum(λ)
    
    return ϕ > 30
end