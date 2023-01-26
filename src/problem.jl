temperature_rate(gas::Gas{N}, c̅ᵥ::N = average_heat_capacity_volume(gas)) where {N<:Number} = -sum(production_rate(s) * internal_energy(gas, s) for s in species(gas)) * inv(density(gas) * c̅ᵥ)
mass_fraction_rates!(Ẏ::AbstractArray{N}, gas::Gas{N}) where {N<:Number} = map!(s -> production_rate(s, Val(:val)) * s.weight / density(gas), Ẏ, species(gas))

"""
RHS function for 0D isochoric problem.
"""
function ZeroDimProblem!(du::Vector{N}, u::Vector{N}, gas::Gas{N}, t::N) where {N<:Real}
    Y = @view u[1:end-1]
    T = last(u)

    ## takes same density each iteration
    TρY!(gas, T, density(gas), Y) |> update

    mass_fraction_rates!(du, gas) ## Ẏ; ∂Y∂t
    du[end] = temperature_rate(gas) ## Ṫ; ∂T∂t
    return nothing
end

function solveZDP(gas::Gas{N}; T::N = temperature(gas), P::N = pressure(gas), Y::Vector{N} = mass_fractions(gas), t∞::N = 5.0,
    maxiters::Int = 100_000, abstol::N = 1e-10, reltol::N = 1e-10, with_IDT = false) where {N<:Real}

    if with_IDT
        saved_values = SavedValues(N, N)
        save_func(u, t, integrator) = last(get_du(integrator))

        steady_state = TerminateSteadyState()
        save_Ṫ = SavingCallback(save_func, saved_values, save_start=false)
        cbs = CallbackSet(steady_state, save_Ṫ)
    else
        cbs = TerminateSteadyState()
    end

    uₒ = vcat(Y, T)
    span = (zero(N), t∞)
    TPY!(gas, T, P, Y)

    syms = [map(s -> s.formula, species(gas)); :T]
    ODEF = ODEFunction(ZeroDimProblem!; syms)
    ODEP = ODEProblem(ODEF, uₒ, span, gas)
    solution = solve(ODEP, CVODE_BDF(), abstol, reltol, maxiters, callback=cbs)

    if with_IDT
        Tₒ = first(solution[:T])
        T∞ = last(solution[:T])

        t, Ṫ = saved_values.t, saved_values.saveval
        Ṫmax, maxind = findmax(Ṫ)

        ΔT = T∞ - Tₒ
        tᵢ = t[maxind]
        tᵣ = ΔT / Ṫmax
        J = IDT(sol, tᵣ)
        return solution, tᵢ, tᵣ, J
    else
        return solution
    end
end

function trapezoid(t::Vector{N}, ƒ::VecOrMat{N}) where {N<:Real}
    m = size(ƒ, 2)
    I = zeros(m)
    for n in Base.OneTo(length(t)-1), p in Base.OneTo(m)
        I[p] += 0.5(t[n+1] - t[n]) * (ƒ[n+1, p] + ƒ[n, p])
    end
    return isone(m) ? only(I) : I
end

function IDT(sol, tᵣ, f = 0.01)
    tₒ, t∞ = (first, last)(sol.t)
    τ = t∞ - tₒ

    t = filter(t -> t > tₒ + f * tᵣ, sol.t)
    tₛ = map(t -> t - f * tᵣ, t)
    
    T = sol(t)[:T]
    Tₛ = sol(tₛ)[:T]

    I = @. (T - Tₛ)^2
    J = trapezoid(t, I) / 2τ
    return J
end