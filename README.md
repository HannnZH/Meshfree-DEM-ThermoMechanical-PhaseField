# A mesh-free deep energy method for phase-field modeling of thermally driven brittle fracture

Reference implementation for the paper of the same name. The displacement and the phase field
are represented by one network and the temperature by a second one. The coordinates enter both
through multiresolution grids of quadratic B-spline features, a positional encoding rather than
a mesh, and every derivative comes from automatic differentiation. Each load step first
minimizes an incremental functional of transient heat conduction and then the thermoelastic
energy of the AT2 phase-field model with the hybrid split, on Monte Carlo integration points
that are redrawn at every iteration and concentrated where the crack is. Essential boundary
conditions hold exactly through lifts, and curved domains enter through an exact NURBS map. On
the quenched slab and annulus the cracks nucleate under the thermal stress, with no crack placed
in advance.

Preprint: https://arxiv.org/abs/2611.17630

Contact: han.zhang7@unsw.edu.au

## Layout

    pfte/          the solver: configuration, geometry and NURBS maps, the temperature network
                   and heat functional, the problems, sampler, energy and training loop, plotting
    examples/      one script per numerical example of the paper; each runs on its own
    fem/           the staggered finite element solvers of the reference solutions

## Requirements

Python 3.10 with the packages in `requirements.txt`. A CUDA device is expected for the
examples, which run in double precision; the code also runs on the CPU, slowly. The finite
element references need `scikit-fem`, and use `pypardiso` for the direct solves if it is
installed.

## Running the examples

Every script writes to the folder given by `--out` (default `runs/<case>`), and `--resume`
continues an interrupted run from its latest checkpoint. The defaults of each script are the
settings of the runs reported in the paper, and every solver option can be overridden on the
command line (`--help` lists them with their defaults).

    python examples/sen_tension.py                      # notched plate in tension, heated by 50 K
    python examples/sen_tension.py --dT_final -50       #   cooled by 50 K
    python examples/sen_tension.py --dT_final 0         #   isothermal

    python examples/sen_shear_fgm.py                    # graded plate in shear, heated by 30 K
    python examples/sen_shear_fgm.py --dT_final -30     #   cooled by 30 K
    python examples/sen_shear_fgm.py --dT_final 0       #   isothermal

    python examples/strip.py                            # quenched glass strip, dT = 165 K
    python examples/strip.py --dT 217                   #   any other temperature jump

    python examples/slab.py                             # quenched slab, 580 K, field 1
    python examples/slab.py --T0 573 --dT_final -280 --het_seed 2    # 280 K
    python examples/slab.py --T0 673 --dT_final -380 --het_seed 3    # 380 K
    python examples/slab.py --T0 773 --dT_final -480 --het_seed 2    # 480 K

    python examples/annulus.py                          # quenched annulus, 580 K
    python examples/annulus.py --T0 573 --dT_final -280              # 280 K, likewise 380, 480 K

Give each variant its own `--out`. The strip of the paper is computed at the jumps 130, 145,
155, 165, 170, 175, 190, 217 and 307 K. On the slab, `--het_seed` selects the realization of
the Weibull field, and the repeats at 380 K add `--seed 1` and `--seed 2`; `--bc_T robin`
replaces the prescribed temperature of the cooled faces by convection. The docstring of each
script states its geometry, material, boundary conditions and loading.

`--seed` sets the network initialization and the streams of integration points. A run repeated
on a different GPU model is not bitwise identical to the original.

Each run writes `config.json`, `train.log`, the load history, the field figures and a
checkpoint per load step.

For orientation, a run of the tension plate took about 5 hours on one V100, of the graded shear
plate about 10 hours, of the slab about 17 hours and of the annulus about 17 hours, and a run of
the strip about 26 hours on one H200.

## Finite element references

The references are staggered phase-field finite element computations (AT2, hybrid split,
bilinear quadrilaterals). Where the neural solver computes the temperature, they solve the
transient conduction on the same clock, and on the strip they impose the same closed-form
profile. Irreversibility is enforced by the history field on the notched plates and by a lower
bound on the phase field on the slab, the annulus and the strip, and the notched-plate
references carry no temperature-dependent toughness (`--gc_tdep 1` adds the law of the neural
solver). `fem/pf_fem.py` computes the notched
plates, `fem/pf_fem_ring.py` the slab and the annulus, on the same realization of the Weibull
field as the neural solver, and `fem/pf_fem_strip.py` the glass strip. The latter two import
their assembly kernels from `fem/pf_fem.py`, so the three files stay in one folder.

    python fem/pf_fem.py --example tension --l 0.01 --refine 9 --stag_max 2000 \
        --dT_final -50 --ramp_rate 2.5e8 --delta_max 8e-4 --nsteps 80
    python fem/pf_fem.py --example shear --split hybrid --fgm --l 0.01 --refine 9 \
        --stag_max 2000 --dT_final 30 --delta_max 3.5e-3 --nsteps 700

    python fem/pf_fem_ring.py --geom slab --h 0.025 --nsteps 100 --bc dirichlet --irr bound \
        --T0 873 --het_seed 1
    python fem/pf_fem_ring.py --h 0.025 --nsteps 100 --bc dirichlet --irr bound \
        --stag_max 1000 --T0 873

    python fem/pf_fem_strip.py --dT 165 --h 0.025 --nsteps 240

The other thermal loads change `--dT_final`, `--T0` (with `--het_seed` on the slab, as above)
or `--dT`. The heated tension plate runs further, `--delta_max 1e-3 --nsteps 100`, and the
isothermal one takes `--dT_final 0` without the ramp. Give each variant its own `--out`.

The finite element references of the notched plates took 2.3 to 2.6 hours (tension) and 4.6 to
6.3 hours (graded shear) on 8 CPU cores.

## Citation

If you use this code in work you publish, please cite the paper:

    @article{zhang2026meshfree,
      author  = {Zhang, Han and Makki Alamdari, Mehrisadat and Shahbodagh, Babak
                 and Vahab, Mohammad and Anitescu, Cosmin and Rabczuk, Timon
                 and Atroshchenko, Elena},
      title   = {A mesh-free deep energy method for phase-field modeling of
                 thermally driven brittle fracture},
      journal = {arXiv preprint arXiv:2611.17630},
      year    = {2026}
    }

## License

MIT, see `LICENSE`.
