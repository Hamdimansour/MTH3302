"""
    roc_instance(y , θ̂ ; u::Real=.5)

Calcul du point associé au seuil 0 < u < 1 de la courbe ROC.

### Arguments
- `y::Array{Int64,1}` : le vecteur des observations composés de 0 et 1.
- `θ̂::Array{Float64,1}` : l'estimation de la probabilité de succès pour chacune des observations.
- `u::Real=.5` : le seuil de décision (1/2 par défaut).

### Details

La fonctionne retourne le taux de vrais positifs et de faux positifs.

### Examples

\```
 julia> p, q = roc_instance(y , θ̂)
 julia> p, q = roc_instance(y , θ̂ ; u=.8)
\```

"""
function roc_instance(y::Array{Int64,1},θ̂::Array{Float64,1} ; u::Real=.5)

    ŷ = zeros(Int64,n)
    ŷ[θ̂.>u] .= 1

    confusionMatrix = zeros(Int64,2,2)
    confusionMatrix[1,1]  = count( (ŷ .== 1) .& (y.== 1) )
    confusionMatrix[2,1]  = count( (ŷ .== 0) .& (y.== 1) )
    confusionMatrix[1,2]  = count( (ŷ .== 1) .& (y.== 0) )
    confusionMatrix[2,2]  = count( (ŷ .== 0) .& (y.== 0) )
    confusionMatrix

    truePositive = confusionMatrix[1,1] / ( confusionMatrix[1,1] + confusionMatrix[2,1] )
    falsePositive = confusionMatrix[1,2] / ( confusionMatrix[1,2] + confusionMatrix[2,2] )

    return truePositive, falsePositive
end


"""
    roc_curve(y , θ̂)

Calcul de la courbe ROC.

### Arguments
- `y::Array{Int64,1}` : le vecteur des observations composés de 0 et 1.
- `θ̂::Array{Float64,1}` : l'estimation de la probabilité de succès pour chacune des observations.

### Details

La fonction retourne un vecteur de taux de vrais positifs et un vecteur taux de faux positifs pour plusieurs valeurs de seuil.

### Examples

\```
 julia> p, q = roc_curve(y , θ̂)
\```

"""
function roc_curve(y::Array{Int64,1}, θ̂::Array{Float64,1})

    u = collect(range(0,stop=1,length=20))

    m = length(u)

    p = zeros(m)
    q = zeros(m)

    for i =1:m
        p[i], q[i] = roc_instance(y, θ̂, u=u[i])
    end

    ind = Int64[]
    q_unique = unique(q)

    for x in q_unique
        push!(ind, findfirst(q .== x))
    end

    q = q[ind]
    p = p[ind]

    perm = sortperm(q)
    p = p[perm]
    q = q[perm]

    return p, q

end


"""
    roc_area(y , θ̂)

Calcul de l'aire sous la courbe ROC par intégration numérique.

### Arguments
- `y::Array{Int64,1}` : le vecteur des observations composés de 0 et 1.
- `θ̂::Array{Float64,1}` : l'estimation de la probabilité de succès pour chacune des observations.

### Details

La fonction retourne l'aire calculée par intégration numérique (méthode des trapèzes).

### Examples

\```
 julia> A = roc_area(y , θ̂)
\```

"""
function roc_area(y::Array{Int64,1}, θ̂::Array{Float64,1})

    p, q = roc_curve(y,θ̂)

    # Intégration numérique de la courbe ROC
    A = sum( (q[i+1]-q[i])*(p[i]+p[i+1])/2 for i=1:length(q)-1)

    return A

end
