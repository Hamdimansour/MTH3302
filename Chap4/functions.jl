"""
    standardize!(X)

Standardisation des colonnes de la matrice (ou vecteur) X.

### Arguments
- `X` : Matrice des données.

### Details

La fonction modifie directement l'argument d'entrée X.

### Examples

\```
 julia> standardize!(X)
\```

"""
function standardize!(X::Array{T,2} where T<:Real)
   
    m = mean(X, dims=1)
    s = std(X, dims=1)
    
    n, p = size(X)
    
    for i=1:n
        for j=1:p
            X[i,j] = (X[i,j] - m[j]) / s[j]
        end
    end
    
    return X
        
end

function standardize!(X::Array{T,1} where T<:Real)
   
    m = mean(X)
    s = std(X)
    
    n = length(X)
    
    for i=1:n
        X[i] = (X[i] - m) ./ s
    end
    
    return X
        
end

"""
    standardize(X)

Standardisation des colonnes de la matrice (ou vecteur) X.

### Arguments
- `X` : Matrice des données.

### Details

La fonction renvoie une copie de la matrice standardisée.

### Examples

\```
 julia> Z = standardize(X)
\```

"""
function standardize(X::Array{T,2} where T<:Real)
   
    m = mean(X, dims=1)
    s = std(X, dims=1)
    
    n, p = size(X)
    
    Z = zeros(n, p)
    
    for i=1:n
        for j=1:p
            Z[i,j] = (X[i,j] - m[j]) / s[j]
        end
    end
    
    return Z
        
end

function standardize(X::Array{T,1} where T<:Real)
   
    m = mean(X)
    s = std(X)
    
    n = length(X)
    
    Z = zeros(n)
    
    for i=1:n
        Z[i] = (X[i] - m) ./ s
    end
    
    return Z
        
end