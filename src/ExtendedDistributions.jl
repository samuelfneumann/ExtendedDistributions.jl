module ExtendedDistributions

using Random
using Reexport
@reexport using Distributions

const _fEPSILON = 1e-6

include("univariates.jl")

export
    ArctanhNormal,
    Metalogistic,
    LogitMetalogistic,
    feasible4,
    feasible3,
    feasible2,
    isfeasible

end
