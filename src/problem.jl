temperature_rate(gas::Gas{N}, c̅ᵥ::N=average_heat_capacity_volume(gas)) where {N<:Real} = -(production_rates(gas) ⋅ internal_energies(gas)) * inv(density(gas) * c̅ᵥ)
mass_fraction_rates!(Ẏ::AbstractArray{N}, gas::Gas{N}) where {N<:Real} = map!((ω̇, w) -> ω̇ * w / density(gas), Ẏ, production_rates(gas), molecular_weights(gas))

"""
RHS function for 0D isochoric problem.
"""
function ZeroDimProblem!(du::Vector{N}, u::Vector{N}, gas::Gas{N}, t::N) where {N<:Real}
    Y = @view u[1:end-1]
    T = last(u)

    ## takes same density each iteration
    TρY!(gas; T, Y) |> update

    mass_fraction_rates!(du, gas) ## Ẏ; dYdt
    du[end] = temperature_rate(gas) ## Ṫ; dTdt
    return nothing
end

function equilibrate(gas::Gas{N}; T::N=temperature(gas), P::N=pressure(gas), Y::Vector{N}=mass_fractions(gas),
    maxiters::Int=100_000, abstol::N=1e-10, reltol::N=1e-13, with_IDT=false) where {N<:Real}

    if with_IDT
        saved_values = SavedValues(N, N)
        save_Ṫ = SavingCallback((u, t, integrator) -> last(get_du(integrator)), saved_values; save_start=false)
        steady_state = TerminateSteadyState()
        callback = CallbackSet(steady_state, save_Ṫ)
    else
        callback = TerminateSteadyState()
    end

    uₒ = vcat(Y, T)
    span = (zero(N), 10.0)
    TPY!(gas; T, P, Y)

    syms = [map(s -> s.formula, species(gas)); :T]
    ODEF = ODEFunction(ZeroDimProblem!; syms)
    ODEP = ODEProblem(ODEF, uₒ, span, gas)
    sol = solve(ODEP, CVODE_BDF(); abstol, reltol, maxiters, callback)

    if with_IDT
        tₒ, t∞ = (first, last)(sol.t)
        Tₒ, T∞ = (first, last)(sol[:T])

        t, Ṫ = saved_values.t, saved_values.saveval
        Ṫmax, maxind = findmax(Ṫ)

        ΔT = T∞ - Tₒ
        tᵢ = t[maxind]
        tᵣ = ΔT / Ṫmax
        J = IDT(sol, tᵣ, tₒ, t∞)
        return sol, tᵣ, J, tᵢ
    else
        return sol
    end
end

function IDT(sol, tᵣ, tₒ, t∞)
    t = filter(t -> t > tₒ + 0.01tᵣ, sol.t)
    tₛ = map(t -> t - 0.01tᵣ, t)

    T = sol(t)[:T]
    Tₛ = sol(tₛ)[:T]

    τ = t∞ - tₒ
    I = @. (T - Tₛ)^2
    J = integrate(t, I) / 2τ
    return J
end