# ZeroDimSteady.jl
Application of `Apophis` on analysis of 0D Steady Combustion

```julia
julia> gas = Gas(:GRI3; T = 1000, P = 1013250, Y = "CH4: 0.05, O2: 0.20, N2: 0.75");

julia> sol = solveZDP(gas);

julia> sol[:T]
266-element Vector{Float64}:
 1000.0
 1000.013518
 1000.035199
 1000.091984
 1000.155837
 1000.342373
 1000.613140
 1000.979599
 1001.142018
 1001.331906
    ⋮
 2703.035058
 2703.045008
 2703.085573
 2703.123614
 2703.241889
 2703.460511
 2703.462647
 2703.461241
 2703.461235
```
