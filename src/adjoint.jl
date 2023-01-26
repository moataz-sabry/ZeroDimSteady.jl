function dẎdY!(dẎdY::AbstractArray{N}, gas::Gas{N}) where {N<:Real}
    #S = species(gas)
    for s₁ in species(gas), s₂ in species(gas) ## (s₁, s₂) in Iterators.product(S, S)
        dẎdY[s₁.k, s₂.k] = production_rate(s₁, Val(:dC))[s₂.k] * s₁.weight / s₂.weight
    end
end

dẎdT!(dẎdT::AbstractArray{N}, gas::Gas{N}) where {N<:Real} = map!(s -> production_rate(s, Val(:dT)) * s.weight / density(gas), dẎdT, species(gas))

function dṪdY!(dṪdY::AbstractArray{N}, gas::Gas{N}, c̅ᵥ::N) where {N<:Real}
    S = species(gas)
    T = temperature(gas)
    Ṫ = temperature_rate(gas, c̅ᵥ)

    map!(dṪdY, S) do s₁
        W₁ = s₁.weight
        cᵥ = heat_capacity_volume(s₁)

        t₀ = inv(W₁ * c̅ᵥ)
        t₁ = -sum((enthalpy(s₂) - Apophis.R * T) * production_rate(s₂, Val(:dC))[s₁.k] for s₂ in S) ## u * ∂ω̇∂C
        t₂ = -Ṫ * cᵥ
        return t₀ * (t₁ + t₂)
    end
    return nothing
end

function dṪdT!(dṪdT::AbstractArray{N}, gas::Gas{N}, c̅ᵥ::N) where {N<:Real}
    T = temperature(gas)
    ρ = density(gas)
    Y = mass_fractions(gas)
    
    S = species(gas)
    Ṫ = temperature_rate(gas, c̅ᵥ)

    dṪdT[] = sum(zip(S, Y)) do (s, y)
        ω̇, dω̇dT = production_rate(s), production_rate(s, Val(:dT))
        u, dudT = (enthalpy(s) - Apophis.R * T), (enthalpy(s, Val(:dT)) - Apophis.R)
        dcᵥdT = heat_capacity_pressure(s, Val(:dT)) ## dcᵥdT = dcₚdT
        tₒ = inv(ρ * c̅ᵥ)

        t₁ = -dω̇dT * u 
        t₂ = -ω̇ * dudT
        t₃ = -Ṫ * y * inv(s.weight * c̅ᵥ) * dcᵥdT
        return tₒ * (t₁ + t₂) + t₃
    end
    return nothing
end

function adjointIDTZDP!(dλ::Matrix{N}, λ::Matrix{N}, (gas, sol, A, tᵣ), t::N) where {N<:Real}
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

    TρY!(gas, T, density(gas), Y) |> update
    update(gas, :dT, :dC)
    c̅ᵥ = average_heat_capacity_volume(gas)

    dẎdY!(dẎdY, gas);     dẎdT!(dẎdT, gas)
    dṪdY!(dṪdY, gas, c̅ᵥ); dṪdT!(dṪdT, gas, c̅ᵥ)

    mul!(dλ, λ, A, -one(N), zero(N)) ## -1.0AB := -λ * A
    dλ[end] = Tₛ - T
    return nothing
end

function solveAdjointIDTZDP(gas::Gas{N}; T::N = temperature(gas), P::N = pressure(gas), Y::Vector{N} = mass_fractions(gas),
    maxiters::Int = 100_000, abstol::N = 1e-8, reltol::N = 1e-8) where {N<:Real}

    sol, tᵢ, tᵣ, J = solveZDP(gas; Y, T, P, with_IDT=true)
    t∞ = last(sol.t)
    
    D = length(species(gas)) + 1
    λₒ = zeros(N, 1, D)
    A = zeros(N, D, D)
    
    p = gas, sol, A, tᵣ
    span = (t∞, tᵣ)
    ODE = ODEProblem(adjointIDTZDP!, λₒ, span, p)

    solAdj = solve(ODE, CVODE_BDF(), abstol, reltol, maxiters)
    return solAdj, J
end

function sensitivity(gas::Gas{<:Real})
    solAdj, J = solveAdjointIGR(gas)
    ƒ = ƒs.saveval
    λ = solAdj

    I = hcat((λ[i] * ƒ[i] for i in eachindex(ƒs.t))...)
    dJdg = trapezoid(I, sol.t)
    return dJdg, J
end