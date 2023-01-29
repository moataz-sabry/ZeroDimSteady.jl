function dẎdY!(dẎdY::AbstractArray{N}, gas::Gas{N}) where {N<:Real}
    ∂ω̇∂C = production_rates(gas, Val(:dC))
    W = molecular_weights(gas)
    K, J = axes(∂ω̇∂C)
    for (k, j) in product(K, J)
        dẎdY[k, j] = ∂ω̇∂C[k, j] * W[k] / W[j]
    end
    return dẎdY
end

dẎdT!(dẎdT::AbstractArray{N}, gas::Gas{N}) where {N<:Real} = map!((ω̇, w) -> ω̇ * w / density(gas), dẎdT, production_rates(gas), molecular_weights(gas))

function dṪdY!(dṪdY::AbstractArray{N}, gas::Gas{N}, c̅ᵥ::N) where {N<:Real}
    W = molecular_weights(gas)
    Ṫ = temperature_rate(gas, c̅ᵥ)

    Cᵥ = heat_capacities_volume(gas)
    U = internal_energies(gas)
    ∂ω̇∂C = production_rates(gas, Val(:dC))
    for (k, (cᵥ, w)) in enumerate(zip(Cᵥ, W))
        t₀ = inv(w * c̅ᵥ)
        t₁ = -sum(u * ∂ω̇∂C[j, k] for (j, u) in enumerate(U))
        t₂ = -Ṫ * cᵥ
        dṪdY[k] = t₀ * (t₁ + t₂)
    end
    return dṪdY
end

function dṪdT!(dṪdT::AbstractArray{N}, gas::Gas{N}, c̅ᵥ::N) where {N<:Real}
    ρ = density(gas)
    Y = mass_fractions(gas)
    
    W = molecular_weights(gas)
    Ṫ = temperature_rate(gas, c̅ᵥ)

    ω̇, dω̇dT = production_rates(gas), production_rates(gas, Val(:dT))
    u, dudT = internal_energies(gas), internal_energies(gas, Val(:dT))
        
    dCᵥdT = heat_capacities_volume(gas, Val(:dT))
    tₒ = inv(ρ * c̅ᵥ)

    t₁ = dω̇dT ⋅ u
    t₂ = ω̇ ⋅ dudT
    t₃ = sum(y * dcᵥdT / w for (y, dcᵥdT, w) in zip(Y, dCᵥdT, W)) * Ṫ / c̅ᵥ
    dṪdT[] = -tₒ * (t₁ + t₂) - t₃
    return dṪdT[]
end

function dṪdA!(gas::Gas{N}, c̅ᵥ::N) where {N<:Real}
    dkfdA = blabla
    dqdkf, dqdkr = kinetics_sensitivity(gas)
    v = stoichiometry_matrix(gas)
    dwdA = v * dqdkf * dkfdA
    return [dẎdA; dṪdA]
end

function adjointZDP!(dλ::Matrix{N}, λ::Matrix{N}, (gas, sol, A, tᵣ), t::N) where {N<:Real}
    u = sol(t)
    Y = @view u[1:end-1]
    T = last(u)
    Tₛ = sol(t - 0.01tᵣ)[end]

    OneToK = eachindex(Y)
    @views begin
        dẎdY = A[OneToK, OneToK]
        dẎdT = A[OneToK, end]
        dṪdY = A[end, OneToK]
    end
    dṪdT = @view A[end, end]

    TρY!(gas, T, density(gas), Y) |> update!
    update!(gas, :dT, :dC)
    c̅ᵥ = average_heat_capacity_volume(gas)

    dẎdY!(dẎdY, gas);     dẎdT!(dẎdT, gas)
    dṪdY!(dṪdY, gas, c̅ᵥ); dṪdT!(dṪdT, gas, c̅ᵥ)

    mul!(dλ, λ, A, -one(N), zero(N)) ## -1.0AB := -λ * A
    dλ[end] = Tₛ - T
    return nothing
end

function solveAdjoint(gas::Gas{N}; T::N = temperature(gas), P::N = pressure(gas), Y::Vector{N} = mass_fractions(gas),
    maxiters::Int = 100_000, abstol::N = 1e-8, reltol::N = 1e-8) where {N<:Real}

    # saved_values = SavedValues(N, Matrix{N})
    # save_func(u, t, integrator) = [(dẎdA, dṪdA)(first(integrator.p))...;]
    # saveg = SavingCallback(save_func, saved_values)

    sol, tᵢ, tᵣ, J = solveZDP(gas; Y, T, P, with_IDT=true)
    t∞ = last(sol.t)
    
    D = length(species(gas)) + 1
    λₒ = zeros(N, 1, D)
    A = zeros(N, D, D)
    
    p = gas, sol, A, tᵣ
    span = (t∞, tᵣ)
    ODE = ODEProblem(adjointZDP!, λₒ, span, p)
    
    solAdj = solve(ODE, CVODE_BDF(), abstol, reltol, maxiters)
    return solAdj, J
end

function sensitivity(gas::Gas{<:Real})
    solAdj, J = solveAdjoint(gas)
    ƒ = ƒs.saveval
    λ = solAdj

    I = hcat((λ[i] * ƒ[i] for i in eachindex(ƒs.t))...)
    dJdg = trapezoid(I, sol.t)
    return dJdg, J
end