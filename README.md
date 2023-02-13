# ZeroDimSteady.jl
`Apophis` application on 0D Steady Combustion

```julia
using ZeroDimSteady
using Unitful: K, atm

gas = Gas(:GRI3; T = 1000K, P = 1atm, Y = "CH4: 0.05, O2: 0.20, N2: 0.75")
sol = equilibrate(gas)
```

![fig](https://user-images.githubusercontent.com/78830303/218363101-b6aba1a7-4feb-4342-aabe-0155f1794d0c.svg)
