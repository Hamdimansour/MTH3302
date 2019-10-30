function GMM(ω::Real, μ₀::Real, σ₀::Real, μ₁::Real, σ₁::Real)
    
    pd = MixtureModel(Normal[ Normal(μ₀, σ₀), Normal(μ₁, σ₁)], [1-ω, ω])
    
    return pd
    
end

function _emstep(pd::MixtureModel,y)
    
    n = length(y)
    
    f₁ = component(pd, 2)
    ω = probs(pd)[2]
    
    lp₁ = log(ω) .+ logpdf.(f₁,y) - logpdf.(pd, y)
    p₁ = exp.(lp₁)
    
    ω̂ = sum(p₁)/n
    
    p₀ = 1 .- p₁
    
    μ̂₀ = sum( p₀.* y) / sum(p₀)
    
    σ̂₀² = sum( p₀.* (y .- μ̂₀).^2 ) / sum(p₀)
    
    μ̂₁ = sum( p₁.* y) / sum(p₁)
    
    σ̂₁² = sum( p₁.* (y .- μ̂₁).^2 ) / sum(p₁)
    
    fd = GMM(ω̂, μ̂₀, sqrt(σ̂₀²), μ̂₁, sqrt(σ̂₁²))
    
    return fd
    
end

function GMMemfit(y::Vector{<:Real} ; initialValue::Vector{<:Real}=Float64[], maxiter::Int=1000, tol::Real=2*eps())
    
    if isempty(initialValue)
        
        ind = (1:n) .< n/2
        
        y₀ = y[ind]
        y₁ = y[.!(ind)]
        
        initialValue = [.5, mean(y₀), std(y₀), mean(y₁), std(y₁)]
        
    end
    
    pd = GMM(initialValue...)
    
    iter = 1
    err = 1
    
    while (err > tol) & (iter < maxiter)
       
        fd = _emstep(pd,y)
        
        err = abs(loglikelihood(fd,y) - loglikelihood(pd,y))
        
        pd = fd
        
        iter +=1
        
    end
    
    if iter>maxiter
        println("Convergence not reached in $maxiter iterations")
    else
        println("Convergence reached in $iter iterations")
    end
    
 return pd
    
end
