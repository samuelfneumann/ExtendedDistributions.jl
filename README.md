RLDist.jl
================
Probability distributions in Julia.

This package re-exports `Distributions.jl` and adds extra distributions. Most
notably, the following extra distributions have been added:

- ArctanhNormal (Squashed Gaussian) Distribution
- Kumaraswamy Distribution

## ToDo
- [ ] Make default floating point precision Float32
- [ ] Some distributions returns Float64 samples even when parameterized by
	Float32's. I've notice this happens in the Gamma distribution, which causes
	it to happen in the Beta distribution as well. Right now, I've just done
	some casting to prevent this, but we should probably just use type-stable
	operations.
