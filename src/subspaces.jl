function samples(P::I, S::I) where {I<:Int} # s: number of parameters & samples
    μ = zeros(P)
    Σ = (1.0)I(P)
    d = MvNormal(μ, Σ)
    x = rand(d, S)
    return x
end

function _sampling(gas::Gas{N}, g::Vector{N}, gₒ::Vector{N}) where {N<:Real}
    pre_exponential_factors!(gas, g)
    dJdg, J = sensitivity(gas)

    dfdx = @. dJdg * g * log(UF[:H2]) / 3J
    pre_exponential_factors!(gas, gₒ)
    return dfdx, J
end

function sampling(gas::Gas{<:Real}, S::Int)
    P = length(reactions(gas)) 
    xₒ = samples(P, S)
    gₒ = pre_exponential_factors(gas)
    G = @. gₒ * exp(1//3 * xₒ * log(UF[:H2]))

    dfdx = zero(G)
    J = zeros(S)
    for (j, gⱼ) in enumerate(eachcol(G))
        dfdx[:, j], J[j] = _sampling(gas, gⱼ, gₒ)
    end
    return dfdx, J, xₒ
end

function monte(gas::Gas{<:Real}, S::Int)
    dfdx, J, xₒ = samplers(gas, S)
    # fLLAM = LLAM(gas, xₒ)

    C = dfdx * dfdx' / s
    λ, W = eigen(C)
    y = W' * xₒ
    return y, W, fLLAM, J, λ, xₒ
end