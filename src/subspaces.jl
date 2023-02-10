function samples(P::N, S::N) where {N<:Int} ## number of parameters & samples
    μ = zeros(P)
    Σ = (1.0)I(P)
    d = MvNormal(μ, Σ)
    x = rand(d, S)
    return x
end

pre_exponential_factors(gas::Gas{<:Real}) = map(r -> r isa Apophis.FallOffReaction ? r.high_pressure_parameters.A : r.forward_rate_parameters.A, reactions(gas))

function change_pre_exponential_factors(gas::Gas{N}, g::AbstractVector{N}) where {N<:Real} ## temporary
    for (i, A) in enumerate(g)
        if gas.mechanism.reactions[i] isa Apophis.FallOffReaction
            @reset gas.mechanism.reactions[i].high_pressure_parameters.A = A
        else
            @reset gas.mechanism.reactions[i].forward_rate_parameters.A = A
        end
    end
    return deepcopy(gas)
end

function _sampling(gas::Gas{N}, UF::Vector{N}, g::AbstractVector{N}) where {N<:Real}
    modified_gas = change_pre_exponential_factors(gas, g)
    dJdg, J = sensitivity(modified_gas)
    dfdx = @. dJdg * g * log(UF) / 3J
    return dfdx, J
end

function sampling(gas::Gas{<:Real}, S::Int)
    P = length(reactions(gas)) 
    UF = rand(1.5:0.25:3.0, P)

    x = samples(P, S)
    gₒ = pre_exponential_factors(gas)
    G = @. gₒ * exp(1//3 * x * log(UF))

    dfdx = zero(G)
    J = zeros(S)
    for (j, gⱼ) in enumerate(eachcol(G))
        dfdx[:, j], J[j] = try _sampling(gas, UF, gⱼ)
        catch
            zeros(P), 0.0
        end
    end
    return dfdx, J, x, G
end

function monte(dfdx::Matrix{<:Real}, x::Matrix{<:Real})
    C = dfdx * dfdx' / size(dfdx, 2)
    λ, W = eigen(C)
    y = W'x
    return y, W, λ
end