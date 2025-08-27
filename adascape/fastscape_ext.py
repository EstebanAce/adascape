import pdb
from fastscape.models import basic_model
from fastscape.processes import SurfaceTopography, UniformRectilinearGrid2D
import numpy as np
import xsimlab as xs
from adascape.base import IR12SpeciationModel
from orographic_precipitation.fastscape_ext import OrographicPrecipitation, OrographicDrainageDischarge


@xs.process
class Speciation:
    """
    Speciation model as a fastscape extension
    """

    trait = xs.index('trait', description="names of the trait(s)")
    init_trait_funcs = xs.group_dict("init_trait_funcs")
    opt_trait_funcs = xs.group_dict("opt_trait_funcs")
    init_abundance = xs.variable(description="initial number of individuals", static=True)
    random_seed = xs.variable(default=None, description="random number generator seed", static=True)
    
    init_x_range_min = xs.variable(default=None, description="min range of individuals on x coordinate", static=True)
    init_x_range_max = xs.variable(default=None, description="max range of individuals on x coordinate", static=True)
    init_y_range_min = xs.variable(default=None, description="min range of individuals on y coordinate", static=True)
    init_y_range_max = xs.variable(default=None, description="max range of individuals on y coordinate", static=True)
    
    abundance = xs.variable(intent="out", description="number of individuals")
    env_field = xs.variable(dims=(('field', "y", "x"), ("y", "x")))

    grid_x = xs.foreign(UniformRectilinearGrid2D, "x")
    grid_y = xs.foreign(UniformRectilinearGrid2D, "y")

    disp_boundary = xs.variable(default=None, description="dispersal boundaries as an xr.DataArray "
                                                          "with vertices [[x,y],...] of bounded area "
                                                          "with dimensions p and d",
                                static=True, dims=[(), ('p', 'd')])

    _model = xs.any_object(description="speciation model instance")
    _individuals = xs.any_object(description="speciation model state dictionary")

    x = xs.on_demand(dims='ind', description="individual's x-position")
    y = xs.on_demand(dims='ind', description="individual's y-position")
    traits = xs.on_demand(dims=('ind', 'trait'), description="individuals' trait values")
    n_offspring = xs.on_demand( dims='ind', description="number of offspring")
    taxon_id = xs.on_demand(dims='ind', description="taxon id number")
    ancestor_id = xs.on_demand(dims='ind', description="ancestor taxa id number", encoding={'fill_value': -1})

    @property
    def individuals(self):
        if self._individuals is None:
            self._individuals = self._model.individuals
        return self._individuals

    @x.compute
    def _get_x(self):
        return self.individuals["x"]

    @y.compute
    def _get_y(self):
        return self.individuals["y"]

    @traits.compute
    def _get_trait(self):
        return self.individuals["trait"]

    @n_offspring.compute
    def _get_n_offspring(self):
        return self.individuals["n_offspring"]

    @taxon_id.compute
    def _get_taxon_id(self):
        return self.individuals["taxon_id"]

    @ancestor_id.compute
    def _get_ancestor_id(self):
        return self.individuals["ancestor_id"]


@xs.process
class IR12Speciation(Speciation):
    """Irwin (2012) Speciation model as a fastscape extension.
    For more info, see :class:`adascape.base.IR12SpeciationModel`.
    """

    sigma_u = xs.variable(default=1, description="trait-mediated competition for a limiting resource, "
                                                 "where the degree of trait similarity is given by "
                                                 "this parameter. If its value is =>1 all individuals "
                                                 "in the local neighbourhood are counted, but if its "
                                                 "values is < 1 then only those individuals with "
                                                 "similar trait values are counted.")
    sigma_f = xs.variable(description="environmental fitness selectivity or width around optimal trait "
                                      "value for each individual's trait")
    sigma_d = xs.variable(description="dispersal variability in meters")
    sigma_m = xs.variable(description="trait variability of mutated offspring")
    p_m = xs.variable(description="mutation probability")
    r = xs.variable(description="fixed neighborhood radius")
    K = xs.variable(description="carrying capacity within a neighborhood")
    taxon_threshold = xs.variable(default=0.05, description="threshold value used in our taxon definition algorithm "
                                                            "to split a group of individuals into two taxon clusters")
    taxon_def = xs.variable(default='traits', description="Taxon definition based on common ancestry and traits ("
                                                          "'traits') or common ancestry, traits and location ("
                                                          "'traits_location')")
    numba_funcs = xs.variable(default=False, description="Use Numba-accelerated functions for taxon ID computation")
    rho = xs.variable(default=0, description="Correlation coefficient between traits, 0 means that traits are "
                                             "independent, where a value of rho different from 0 and "
                                             "between -1 and 1, will determine the degree of correlation between "
                                             "traits for all individuals")

    fitness = xs.on_demand(
        dims='ind',
        description="individual's fitness value"
    )

    def _get_model_params(self):
        return {
            "r": self.r,
            "K": self.K,
            "sigma_d": self.sigma_d,
            "sigma_m": self.sigma_m,
            "sigma_f": self.sigma_f,
            "p_m": self.p_m,
            "random_seed": self.random_seed,
            "taxon_threshold": self.taxon_threshold,
            "rho": self.rho,
            "sigma_u": self.sigma_u,
            "numba_funcs": self.numba_funcs,
            "taxon_def": self.taxon_def,
        }

    def initialize(self):
        X, Y = np.meshgrid(self.grid_x, self.grid_y)

        trait_names = [k[0] for k in self.init_trait_funcs]
        self.trait = np.array(trait_names, dtype="S")

        self._model = IR12SpeciationModel(
            X, Y, self.init_trait_funcs, self.opt_trait_funcs, self.init_abundance,
            always_direct_parent=False,
            **self._get_model_params()
        )
        if self.init_x_range_min is None and self.init_x_range_max is None:
            init_x_range = None
        else:
            init_x_range = (self.init_x_range_min, self.init_x_range_max)

        if self.init_y_range_min is None and self.init_y_range_max is None:
            init_y_range = None
        else:
            init_y_range = (self.init_y_range_min, self.init_y_range_max)

        self._model.initialize(init_x_range, init_y_range)

    def run_step(self):
        # reset individuals "cache"
        self._individuals = None

        # maybe update model parameters
        self._model.params.update(self._get_model_params())

        self.abundance = self._model.abundance
        self._model.evaluate_fitness()

    @xs.runtime(args='step_delta')
    def finalize_step(self, dt):
        self._model.update_individuals(dt, self.disp_boundary)

    @fitness.compute
    def _get_fitness(self):
        return self.individuals["fitness"]


@xs.process
class CompoundEnvironment:
    """Multiple environment fields defined on the same grid.
    """
    field_arrays = xs.group_dict("env_field")
    env_field = xs.foreign(Speciation, "env_field", intent="out")

    def initialize(self):
        self.env_field = np.stack(list(self.field_arrays.values()))

    def run_step(self):
        self.env_field = np.stack(list(self.field_arrays.values()))


@xs.process
class ElevationEnvField:
    """Topographic elevation used as one environment field for the
    speciation model.

    """
    elevation = xs.foreign(SurfaceTopography, "elevation")
    field = xs.variable(dims=("y", "x"), intent="out", groups="env_field")

    def initialize(self):
        self.field = self.elevation

    def run_step(self):
        self.field = self.elevation


@xs.process
class PrecipitationField:
    """
    Orographic precipitation used as an environmental field
    for the speciation model.
    """
    precip = xs.foreign(OrographicPrecipitation, 'precip_rate')
    field = xs.variable(dims=("y", "x"), intent="out", groups="env_field")

    def initialize(self):
        self.field = self.precip

    def run_step(self):
        self.field = self.precip


@xs.process
class RandomSeedFederation:
    """One random seed to rule them all!"""

    seed = xs.variable(
        default=None,
        description="random number generator seed",
        static=True
    )

    the_seed = xs.variable(intent='out', global_name="random_seed")

    def initialize(self):
        self.the_seed = self.seed


@xs.process
class TraitBase:
    """Base class for representing a single trait.

    Do not use this class directly in a xsimlab.Model. Instead,
    create one subclass of this class for one trait.

    Every subclass must at least provide an implementation for
    computing optimal trait values.

    For convenience, this class contains a default implementation
    for computing initial trait values, although it could be
    overriden in subclasses.

    """
    random_seed = xs.variable(
        default=None,
        description="random number generator seed",
        static=True
    )
    init_trait_min = xs.variable(default=0, description="min initial trait value")
    init_trait_max = xs.variable(default=1, description="max initial trait value")

    init_trait_func = xs.any_object(
        description="initialize trait function", groups="init_trait_funcs"
    )
    opt_trait_func = xs.any_object(
        description="optimal trait function", groups="opt_trait_funcs"
    )

    def _compute_init_trait(self, init_abundance):
        """Set initial trait values for individuals.

        By default, initial values are randomly generated from
        a Uniform distribution bounded by
        `init_value_min` and `init_value_max`.

        It is possible to provide alternative implementation in
        subclasses, though.

        Parameters
        ----------
        init_abundance : int
            Number of individuals for the initial population.

        Returns
        -------
        init_trait_values : array
            Initial trait value of each individual.

        """
        return self._rng.uniform(
            self.init_trait_min, self.init_trait_max, init_abundance
        )

    def _compute_opt_trait(self, grid_positions):
        """This is where the computation of the optimal trait
        must be implemented in subclasses.

        Parameters
        ----------
        grid_positions : array-like
            Positions of individuals aligned on the model grid,
            which may be used to retrieve local environment values
            for each individual.

        Returns
        -------
        opt_trait : array
            Optimal trait value for each individual.

        """
        # must be implemented in subclasses
        raise NotImplementedError

    def initialize(self):
        self._rng = np.random.default_rng(self.random_seed)
        self.init_trait_func = self._compute_init_trait
        self.opt_trait_func = self._compute_opt_trait


@xs.process
class FastscapeElevationTrait(TraitBase):
    """Example of a trait that is based on the elevation of the
    topographic surface simulated by Fastscape.

    This process computes normalized optimal trait values
    that linearly depend on elevation.
    """
    topo_elevation = xs.foreign(SurfaceTopography, "elevation")
    random_seed = xs.foreign(RandomSeedFederation, 'seed', intent='in')

    lin_slope = xs.variable(
        description="slope of opt. trait vs. elevation linear relationship"
    )
    norm_min = xs.variable(
        description="min elevation value for normalization"
    )
    norm_max = xs.variable(
        description="max elevation value for normalization"
    )

    def _compute_opt_trait(self, grid_positions):
        env_field = self.topo_elevation.ravel()[grid_positions]
        norm_env_field = (env_field - self.norm_min) / (self.norm_max - self.norm_min)
        opt_trait = ((self.lin_slope * (norm_env_field - 0.5)) + 0.5)

        return opt_trait


@xs.process
class FastscapePrecipitationTrait(TraitBase):
    """Example of a trait that is based on the precipitation over
    topographic surface simulated by Fastscape.

    This process computes normalized optimal trait values
    that linearly depend on precipitation.
    """
    oro_precipitation = xs.foreign(OrographicPrecipitation, "precip_rate")
    random_seed = xs.foreign(RandomSeedFederation, 'seed', intent='in')

    lin_slope = xs.variable(
        description="slope of opt. trait vs. precipitation linear relationship"
    )
    norm_min = xs.variable(
        description="min precipitation value for normalization"
    )
    norm_max = xs.variable(
        description="max precipitation value for normalization"
    )

    def _compute_opt_trait(self, grid_positions):
        env_field = self.oro_precipitation.ravel()[grid_positions]
        norm_env_field = (env_field - self.norm_min) / (self.norm_max - self.norm_min)
        opt_trait = ((self.lin_slope * (norm_env_field - 0.5)) + 0.5)

        return opt_trait


adascape_IR12_model = basic_model.update_processes(
    {'life': IR12Speciation,
     'trait_elev': FastscapeElevationTrait,
     'trait_prep': FastscapePrecipitationTrait,
     'life_env': CompoundEnvironment,
     'elev_field': ElevationEnvField,
     'prec_field': PrecipitationField,
     'random': RandomSeedFederation,
     'orographic': OrographicPrecipitation,
     'drainage': OrographicDrainageDischarge}
)
