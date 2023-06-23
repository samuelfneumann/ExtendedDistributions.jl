const discrete_distributions = []

const continuous_distributions = [
    "arctanhnormal",
    "metalogistic",
    "logitmetalogistic",
    # "kumaraswamy",
]

for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end

for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end
