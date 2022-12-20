import pdb
import textwrap
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial as spatial
import scipy.spatial.distance as dist
from scipy.cluster.vq import kmeans2


class SpeciationModelBase:
    """
    Speciation Model base class with common methods for the different
    types of speciation models.
    """

    def __init__(self, grid_x, grid_y, init_trait_funcs, opt_trait_funcs, init_abundance,
                 lifespan=None, random_seed=None, rescale_rates=False, always_direct_parent=True,
                 on_extinction='warn', taxon_threshold=0.05, taxon_def='traits', rho=0):
        """
        Initialization of based model.

        Parameters
        ----------
        grid_x : array-like
            Grid x-coordinates.
        grid_y : array_like
            Grid y-coordinates.
        init_trait_funcs : dict
            with callables to generate initial values of each trait for each individual.
        opt_trait_funcs : dict
            with callables to compute optimal trait values of each trait for each individual.
        init_abundance : int
            Total number of individuals generated as the initial population.
        lifespan : float, optional
            Reproductive lifespan of organism. If None (default), the
            lifespan will always match time step length
        random_seed : int or :class:`numpy.random.default_rng` object
            Fixed random state for reproducible experiments.
            If None (default), results will differ from one run
            to another.
        rescale_rates : bool
            If True (default) rates and parameters will be rescaled as
            a fraction of the square root number of generations per
            time step.
        always_direct_parent : bool, optional
            If True (default), the id of the parent set for each individual
            of the current population will always correspond to its direct
            parent. If False, those id values may correspond to older
            ancestors. Set this parameter to False if you want to preserve the
            connectivity of the generation tree built by calling
            ``.population`` or ``.to_dataframe()`` at arbitrary steps of a
            model run.
        on_extinction : {'warn', 'raise', 'ignore'}
            Behavior when no offspring is generated (total extinction of
            population) during model runtime. 'warn' (default) displays
            a RuntimeWarning, 'raise' raises a RuntimeError (the simulation
            stops) or 'ignore' silently continues the simulation
            doing nothing (no population).
        taxon_threshold : float, optional
            distance threshold to construct clusters.
            default  = 0.05
        taxon_def: {'traits', 'traits_location'}
                   Variables use to define a taxon either to be based on individual's traits
                   and their shared common ancestry ('trait'), or the same as before but
                   also considering individuals location ('traits_location'), default 'traits'
        rho:  float
            correlation coefficient between traits. Default = 0 all traits are
             independent of each other
        """
        valid_on_extinction = ('warn', 'raise', 'ignore')

        if on_extinction not in valid_on_extinction:
            raise ValueError(
                "invalid value found for 'on_extinction' parameter. "
                "Found {!r}, must be one of {!r}"
                    .format(on_extinction, valid_on_extinction)
            )

        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        self._grid_bounds = {'x': np.array([grid_x.min(), grid_x.max()]),
                             'y': np.array([grid_y.min(), grid_y.max()])}
        self._grid_index = self._build_grid_index([grid_x, grid_y])
        self._individuals = {}
        self._init_abundance = init_abundance
        self._rng = np.random.default_rng(random_seed)

        # https://stackoverflow.com/questions/16016959/scipy-stats-seed
        self._truncnorm = stats.truncnorm
        self._truncnorm.random_state = self._rng

        self._params = {
            'lifespan': lifespan,
            'random_seed': random_seed,
            'always_direct_parent': always_direct_parent,
            'on_extinction': on_extinction,
            'taxon_threshold': taxon_threshold,
            'taxon_def': taxon_def,
            'rho': rho
        }
        self._env_field_bounds = None
        self._rescale_rates = rescale_rates
        self._set_direct_parent = True

        # dict of callables to generate initial values for each trait
        self._init_trait_funcs = init_trait_funcs
        # dict of callables to compute optimal values for each trait
        self._opt_trait_funcs = opt_trait_funcs

        # number of traits
        self.n_traits = len(self._init_trait_funcs)
        assert len(self._init_trait_funcs) == len(self._opt_trait_funcs)

        # for test of taxon definition
        valid_taxon_def = ('traits', 'traits_location')
        if taxon_def not in valid_taxon_def:
            raise ValueError(
                "invalid value found for 'taxon_def' parameter. "
                "Found {!r}, must be one of {!r}"
                    .format(taxon_def, valid_taxon_def)
            )

    def initialize(self, x_range=None, y_range=None):
        """
        Initialization of a group of individuals with randomly distributed traits,
        and which are randomly located in a two-dimensional grid.

        Parameters
        ----------
        x_range : tuple, optional
            Spatial range (min, max) to define initial spatial bounds
            of population in the x direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the x direction.
        y_range : tuple, optional
            Spatial range (min, max) to define initial spatial bounds
            of population in the y direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the y direction.
        """

        x_bounds = self._grid_bounds['x']
        y_bounds = self._grid_bounds['y']
        x_range = x_range or x_bounds
        y_range = y_range or y_bounds

        if ((x_range[0] < x_bounds[0]) or (x_range[1] > x_bounds[1]) or
                (y_range[0] < y_bounds[0]) or (y_range[1] > y_bounds[1])):
            raise ValueError("x_range and y_range must be within model bounds")

        # array of shape (n_individuals, n_traits)
        init_traits = np.column_stack(
            [func(self._init_abundance) for func in self._init_trait_funcs.values()]
        )

        clus = self._spect_clus(init_traits,
                                taxon_threshold=self._params['taxon_threshold'])
        taxon_id = clus + 1
        ancestor_id = clus.copy()

        init_population = {'step': 0,
                           'time': 0.,
                           'dt': 0.,
                           'x': self._sample_in_range(x_range),
                           'y': self._sample_in_range(y_range),
                           'trait': init_traits,
                           'taxon_id': taxon_id,
                           'ancestor_id': ancestor_id,
                           'n_offspring': np.zeros(init_traits.shape[0])
                           }
        self._individuals.update(init_population)

    def _compute_taxon_ids(self):
        """
        Method to define taxa based on individual's traits and their shared common ancestry
        using 1) a hierarchical clustering algorithm from scipy.spatial.hierarchy or
        2) spectral clustering algorithm. For hierarchical clustering a distance value
        and distance metric needs to be specified,the latter by default is set to 'ward' distance.
        For spectral clustering only distance value needs to be specified.

        Returns
        -------
        taxon_ids based on the clustering of individuals with similar trait values
        and common ancestry.
        """
        if self._set_direct_parent:
            new_id_key = 'taxon_id'
        else:
            new_id_key = 'ancestor_id'
        current_ancestor_id = np.repeat(self._individuals[new_id_key], self._individuals['n_offspring'].astype('int'))
        max_clus = self._individuals['taxon_id'].max()

        new_taxon_id = np.zeros_like(current_ancestor_id)
        for ans in np.unique(current_ancestor_id):
            ans_indx = np.where(current_ancestor_id == ans)[0]
            clusdata = pd.DataFrame(self._individuals['trait'][ans_indx])
            if self.params['taxon_threshold'] == 'traits_location':
                clusdata['x'] = self._individuals['x'][ans_indx] / self._grid_bounds['x'][1]
                clusdata['y'] = self._individuals['y'][ans_indx] / self._grid_bounds['y'][1]
            clus = self._spect_clus(clusdata,
                                    taxon_threshold=self._params['taxon_threshold'])
            if max_clus < np.max(clus):
                new_clus = clus + 1
                new_taxon_id[ans_indx] = new_clus.astype(int)
            elif max_clus >= np.max(clus):
                new_clus = clus + 1 + max_clus
                new_taxon_id[ans_indx] = new_clus.astype(int)
            max_clus = np.max(new_clus).astype(int)

        return new_taxon_id, current_ancestor_id

    def _spect_clus(self, clus_data, taxon_threshold=0.05, split_size=10):
        """
        Spectral clustering algorithm based on von Luxburg (2007), which we modified to:
            1) only divide groups of individuals larger than "split_size",
            2) if division occurs only divide into a maximum of two clusters or taxon_ids,
            3) the minimum size of the divided clusters or taxon_ids must be half of split_size.

        Ulrike von Luxburg (2007) A tutorial on spectral clustering.Statistics and Computing,
        17(4):395-416. doi:10.1007/s11222-007-9033-z

        Parameters
        ----------
        clus_data : array-like
                    individual's trait data to be used in the clustering
        split_size : int
                    minimum number of individuals (observations) in cluster data to perform the clustering algorithm

        Returns
        -------
        array-like of int
            with taxon id

        """
        try:
            D2Mat = dist.squareform(dist.pdist(clus_data))
        except:
            D2Mat = dist.squareform(dist.pdist(clus_data[:, np.newaxis]))

        if clus_data.shape[0] > split_size:

            W = np.exp(-np.abs(D2Mat) ** 2 / (2 * taxon_threshold ** 2))  # Gaussian similarity
            W[D2Mat > taxon_threshold] = 0

            D = np.diag(np.sum(W, axis=1))
            L = D - W  # L is a real symmetric (see Proposition 1 Luxburg 2007)
            E, U = np.linalg.eigh(L)
            E = np.real(E)  # remove tiny imaginary numbers
            # (should be tiny because L is positive semi-definite by Theorem)
            U = np.real(U)  # remove tiny imaginary numbers
            # (should be because for real symmetric matrices, U is an orthonormal matrix)
            E[np.isclose(E, np.zeros(len(E)))] = 0  # remove tiny numbers that are too close to zeros
            n_comp = np.sum(E == 0)
            if n_comp > 1:
                k = min(2, int(n_comp))
                centroid, labels = kmeans2(U[:, :k], k=k, minit='points', seed=self._rng)
                count1 = np.sum(labels == 0)
                count2 = np.sum(labels == 1)
                if count1 < split_size // 2:
                    labels[labels == 0] = 1
                if count2 < split_size // 2:
                    labels[labels == 1] = 0
                if np.all(labels.astype(int) == 1):
                    labels = np.zeros(clus_data.shape[0])
            else:
                labels = np.zeros(clus_data.shape[0])
        else:
            labels = np.zeros(clus_data.shape[0])
        return labels

    @property
    def params(self):
        """Model parameters.

        Returns
        -------
        dict
            Model parameters
        """
        return self._params

    @property
    def individuals(self):
        """Individuals' data at the current time step.

        Returns
        -------
        dict
            Individuals data
        """
        self._set_direct_parent = True
        return self._individuals

    @property
    def abundance(self):
        """Number of individuals at the current time
        step.

        Returns
        -------
        int or None
            Number of individuals
            (return None if a group of individuals has not yet been initialized)
        """
        if not self._individuals:
            return None
        else:
            return self._individuals['trait'][:, 0].size

    def to_dataframe(self, varnames=None):
        """Individuals data at the current time step as a
        pandas Dataframe.

        Parameters
        ----------
        varnames : list or string, optional
            Only export those variable name(s) as dataframe column(s).
            Default: export all variables.

        Returns
        -------
        pandas.Dataframe
            Individuals data
        """

        individuals_data = self._individuals.copy()
        for i in range(self._individuals['trait'].shape[1]):
            individuals_data['trait_' + str(i)] = individuals_data['trait'][:, i]

        individuals_data.pop('trait')

        if varnames is None:
            data = individuals_data
        elif isinstance(varnames, str):
            data = {varnames: individuals_data[varnames]}
        else:
            data = {k: individuals_data[k] for k in varnames}
        return pd.DataFrame(data)

    @staticmethod
    def _build_grid_index(grid_coords):
        """
        Builds scipy kd-tre for a quick indexing of all
        points in a grid.

        Parameters
        ----------
        grid_coords : list of arrays
            x and y points in the grid

        Returns
        -------
        scipy.spatial.cKDTree
            kd-tree index object for the set of grid points
        """
        grid_points = np.column_stack([c.ravel() for c in grid_coords])
        return spatial.cKDTree(grid_points)

    def _sample_in_range(self, values_range):
        """
        Draw a random sample of values for a given range following
        a uniform distribution.

        Parameters
        ----------
        values_range : list or tuple
            max and min value from with to draw random values

        Returns
        -------
        array_like
            random sample of values between given range
        """
        return self._rng.uniform(values_range[0], values_range[1], self._init_abundance)

    def _mov_within_bounds(self, x, y, sigma):
        """
        Move and check if the location of individuals are within grid range.

        Parameters
        ----------
        x : array_like
            locations along the x coordinate
        y : array_like
            locations along the y coordinate
        sigma : float
            movement variability
        Returns
        -------
        array-like
            new coordinate for the moved individuals.
        """
        # TODO: check effects of movement and boundary conditions
        # TODO: Make boundary conditions of speciation model to match those of LEM
        x = x / self._grid_bounds['x'][1]
        y = y / self._grid_bounds['y'][1]
        new_x = self._truncnorm.rvs(a=(0 - x) / sigma, b=(1 - x) / sigma, loc=x, scale=sigma)
        new_y = self._truncnorm.rvs(a=(0 - y) / sigma, b=(1 - y) / sigma, loc=y, scale=sigma)
        return new_x*self._grid_bounds['x'][1], new_y*self._grid_bounds['y'][1]

    def _mutate_trait(self, trait, sigma):
        """
        Mutate individual trait values within a range between 0 and 1

        Parameters
        ----------
        trait: array_like
               trait values
        sigma: float
               trait variability
        Returns
        -------
        array-like
            Mutate trait values
        """
        a, b = (0 - trait) / sigma, (1 - trait) / sigma
        mut_trait = self._truncnorm.rvs(a, b, loc=trait, scale=sigma)
        return mut_trait

    def _scaled_param(self, param, dt):
        """ Rescale a parameter as a fraction of the square root
            of the number of generations per time step.
        param : float
            parameter value.
        dt : float
            time step.
        """
        if self._params['lifespan'] is None:
            n_gen = 1.
        else:
            n_gen = dt / self._params['lifespan']

        return param / np.sqrt(n_gen)

    def _update_individuals(self, dt):
        """Require implementation in subclasses."""
        raise NotImplementedError()

    def update_individuals(self, dt):
        """Update individuals' data (generate, mutate, and disperse).

        Parameters
        ----------
        dt : float
            Time step duration.

        """
        self._update_individuals(dt)

        if not self._params['always_direct_parent']:
            self._set_direct_parent = False

    def __repr__(self):
        class_str = type(self).__name__
        population_str = "individuals: {}".format(
            self.abundance or 'not initialized')
        params_str = "\n".join(["{}: {}".format(k, v)
                                for k, v in self._params.items()])

        return "<{} ({})>\nParameters:\n{}\n".format(
            class_str, population_str, textwrap.indent(params_str, '    '))


class IR12SpeciationModel(SpeciationModelBase):
    """Model of speciation along an environmental gradient defined on a
    2-d grid.

    This model is adapted from:

    Irwin D.E., 2012. Local Adaptation along Smooth Ecological
    Gradients Causes Phylogeographic Breaks and Phenotypic Clustering.
    The American Naturalist Vol. 180, No. 1, pp. 35-49.
    DOI: 10.1086/666002

    A model run starts with a given number of individuals with random
    positions (x, y) generated uniformly within the grid bounds and
    initial "trait" values generated uniformly within a given range.

    Then, at each step, the number of offspring for each individual is
    determined using a fitness value computed from the comparison of
    environmental ("trait" vs. "optimal trait") and population density
    ("number of individuals in the neighborhood" vs "capacity")
    variables measured locally. Environmental variables are given on a
    grid, while the neighborhood is defined by a circle of a given
    radius centered on each individual.

    New individuals are generated from the offspring, which undergo
    some random dispersion (position) - and mutation (trait
    value). Dispersion is constrained so that all individuals stay
    within the domain delineated by the grid.
    """

    def __init__(self, grid_x, grid_y, init_trait_funcs, opt_trait_funcs, init_abundance,
                 lifespan=None, random_seed=None, always_direct_parent=True,
                 on_extinction='warn', taxon_threshold=0.05,
                 nb_radius=500., car_cap=1000., sigma_env_trait=0.3, sigma_mov=5.,
                 sigma_mut=0.05, mut_prob=0.05, taxon_def='traits', rho=0):
        """Initialization of speciation model without competition.

        Parameters
        ----------
        nb_radius: float
            Fixed radius of the circles that define the neighborhood
            around each individual.
        car_cap: int
            Carrying capacity of group of individuals within the neighborhood area.
        sigma_env_trait: float
            Width of fitness curve.
        sigma_mov: float
            Width of dispersal curve.
        sigma_mut: float
            Width of mutation curve.
        mut_prob: float
            Probability of mutation occurring in offspring.
        """
        super().__init__(grid_x=grid_x, grid_y=grid_y,
                         init_trait_funcs=init_trait_funcs,
                         opt_trait_funcs=opt_trait_funcs,
                         init_abundance=init_abundance,
                         lifespan=lifespan,
                         random_seed=random_seed,
                         always_direct_parent=always_direct_parent,
                         on_extinction=on_extinction,
                         taxon_threshold=taxon_threshold,
                         taxon_def=taxon_def,
                         rho=rho)

        # default parameter values
        self._params.update({
            'nb_radius': nb_radius,
            'car_cap': car_cap,
            'sigma_env_trait': sigma_env_trait,
            'sigma_mov': sigma_mov,
            'sigma_mut': sigma_mut,
            'mut_prob': mut_prob
        })

    def _get_n_gen(self, dt):
        """
        Number of generations during one time step.

        Parameters
        ----------
        dt : float
            Time step duration.

        Returns
        -------
        float
            number of generations per time step.
        """

        if self._params['lifespan'] is None:
            return 1.
        else:
            return dt / self._params['lifespan']

    def _count_neighbors(self, pop_points):
        """
        count number of neighbouring individual in a given radius.

        Parameters
        ----------
        pop_points : list of array
            location of individuals in a grid.

        Returns
        -------
        array-like
            number of neighbouring individual in a give radius.

        """
        index = spatial.cKDTree(pop_points)
        neighbors = index.query_ball_tree(index, self._params['nb_radius'])

        return np.array([len(nb) for nb in neighbors])

    def evaluate_fitness(self, dt):
        """Evaluate fitness and generate offspring number for group of individuals and
        with environmental conditions both taken at the current time step.

        Parameters
        ----------
        dt : float
            Time step duration.

        """
        if self._rescale_rates:
            sigma_env_trait = self._scaled_param(self._params['sigma_env_trait'], dt)
        else:
            sigma_env_trait = self._params['sigma_env_trait']

        if self.abundance:
            individual_positions = np.column_stack([self._individuals['x'], self._individuals['y']])
            _, grid_positions = self._grid_index.query(individual_positions)

            # array of shape (n_individuals, n_traits)
            opt_trait = np.column_stack(
                [func(grid_positions) for func in self._opt_trait_funcs.values()]
            )

            # compute offspring sizes
            r_d = self._params['car_cap'] / self._count_neighbors(individual_positions)

            delta_trait_i = self._individuals['trait'] - opt_trait
            diag_val = np.ones(self._individuals['trait'].shape[1]) * sigma_env_trait
            sigma_diag = np.diag(diag_val ** 2)
            sigma_offdiag = self._params['rho'] * (diag_val[:, np.newaxis] * diag_val[np.newaxis, :] - sigma_diag)
            sigma_i = sigma_diag + sigma_offdiag
            fitness = np.zeros(self._individuals['trait'].shape[0])
            inv_sigma = np.linalg.inv(sigma_i)
            for i in range(self._individuals['trait'].shape[0]):
                fitness[i] = np.exp(-1 / 2 * np.dot(delta_trait_i[i], np.dot(inv_sigma, delta_trait_i[i].T)))

            n_gen = self._get_n_gen(dt)
            n_offspring = np.round(r_d * fitness * np.sqrt(n_gen)).astype('int')

        else:
            fitness = np.zeros(self._individuals['trait'].shape[1])
            n_offspring = np.zeros(self._individuals['trait'].shape[1])

        self._individuals.update({
            'fitness': fitness,
            'n_offspring': n_offspring
        })

    def _update_individuals(self, dt):
        """Update individuals' data (generate, mutate, and disperse).

        Parameters
        ----------
        dt : float
            Time step duration.
        """
        if self._rescale_rates:
            mut_prob = self._scaled_param(self._params['mut_prob'], dt)
            sigma_mov = self._scaled_param(self._params['sigma_mov'], dt)
            sigma_mut = self._scaled_param(self._params['sigma_mut'], dt)
        else:
            mut_prob = self._params['mut_prob']
            sigma_mov = self._params['sigma_mov']
            sigma_mut = self._params['sigma_mut']

        n_offspring = self._individuals['n_offspring']

        if not n_offspring.sum():
            # population total extinction
            if self._params['on_extinction'] == 'raise':
                raise RuntimeError("no offspring generated. "
                                   "Model execution has stopped.")

            if self._params['on_extinction'] == 'warn':
                warnings.warn("no offspring generated. "
                              "Model execution continues with no population.",
                              RuntimeWarning)

            new_individuals = {k: np.array([])
                               for k in ('x', 'y', 'trait')}
            new_individuals['trait'] = np.expand_dims(new_individuals['trait'], 1)
        else:
            # generate offspring
            new_individuals = {k: np.repeat(self._individuals[k], n_offspring)
                               for k in ('x', 'y')}
            new_individuals['trait'] = np.repeat(self._individuals['trait'], n_offspring, axis=0)

            # mutate offspring
            to_mutate = self._rng.uniform(0, 1, new_individuals['trait'].shape[0]) < mut_prob
            for i in range(new_individuals['trait'].shape[1]):
                new_individuals['trait'][:, i] = np.where(to_mutate,
                                                          self._mutate_trait(new_individuals['trait'][:, i], sigma_mut),
                                                          new_individuals['trait'][:, i])

            # disperse offspring within grid bounds
            new_x, new_y = self._mov_within_bounds(new_individuals['x'],
                                                   new_individuals['y'],
                                                   sigma_mov)
            new_individuals['x'] = new_x
            new_individuals['y'] = new_y

        self._individuals['step'] += 1
        self._individuals['time'] += dt
        self._individuals.update(new_individuals)
        if not n_offspring.sum():
            taxon_id, ancestor_id = np.array([]), np.array([])
        else:
            taxon_id, ancestor_id = self._compute_taxon_ids()
        self._individuals.update({'taxon_id': taxon_id})
        self._individuals.update({'ancestor_id': ancestor_id})

        # reset fitness / offspring data
        self._individuals.update({
            'fitness': np.zeros(self._individuals['trait'].shape[0]),
            'n_offspring': np.zeros(self._individuals['trait'].shape[0])
        })


class DD03SpeciationModel(SpeciationModelBase):
    """
    Speciation model for asexual populations adapted from:
        Doebeli, M., & Dieckmann, U. (2003).
        Speciation along environmental gradients.
        Nature, 421, 259–264.
        https://doi.org/10.1038/nature01312.Published.
    """

    def __init__(self, grid_x, grid_y, init_trait_funcs, opt_trait_funcs, init_abundance,
                 lifespan=None, random_seed=None, always_direct_parent=True,
                 on_extinction='warn', taxon_threshold=0.05,
                 birth_rate=1, movement_rate=5, car_cap_max=500, sigma_env_trait=0.3,
                 mut_prob=0.005, sigma_mut=0.05, sigma_mov=0.12, sigma_comp_trait=0.9,
                 sigma_comp_dist=0.19, taxon_def='traits', rho=0):
        """
        Initialization of speciation model with competition.

        Parameters
        ----------
        birth_rate : integer or float
            birth rate of individuals
        movement_rate : integer of float
            movement/dispersion rate of individuals
        car_cap_max : integer
            maximum carrying capacity
        sigma_env_trait : float
            variability of trait-environment relationship
        mut_prob : float
            mutation probability
        sigma_mut : float
            variability of mutated trait
        sigma_mov : float
            variability of movement distance
        sigma_comp_trait : float
            competition variability for trait distance between individuals
        sigma_comp_dist : float
            competition variability for spatial distance between individuals
        """

        super().__init__(grid_x=grid_x, grid_y=grid_y,
                         init_trait_funcs=init_trait_funcs,
                         opt_trait_funcs=opt_trait_funcs,
                         init_abundance=init_abundance,
                         lifespan=lifespan,
                         random_seed=random_seed,
                         always_direct_parent=always_direct_parent,
                         on_extinction=on_extinction,
                         taxon_threshold=taxon_threshold,
                         taxon_def=taxon_def,
                         rho=rho)

        self._params.update({
            'birth_rate': birth_rate,
            'movement_rate': movement_rate,
            'car_cap_max': car_cap_max,
            'sigma_env_trait': sigma_env_trait,
            'mut_prob': mut_prob,
            'sigma_mut': sigma_mut,
            'sigma_mov': sigma_mov,
            'sigma_comp_trait': sigma_comp_trait,
            'sigma_comp_dist': sigma_comp_dist,
        })

    def evaluate_fitness(self, dt):
        """
        Evaluate fitness of individuals' in a given environmental field.
        The computation is based on the Gillespie algorithm for a group
        of individuals that randomly grows, moves, and dies.

        Parameters
        ----------
        dt : float
            Time step duration.

        """
        if self.abundance:
            individual_positions = np.column_stack([self._individuals['x'], self._individuals['y']])
            _, grid_positions = self._grid_index.query(individual_positions)

            # array of shape (n_individuals, n_traits)
            opt_trait = np.column_stack(
                [func(grid_positions) for func in self._opt_trait_funcs.values()]
            )

            # Compute events probabilities
            if self._rescale_rates:
                sigma_env_trait = self._scaled_param(self._params['sigma_env_trait'], dt)
                sigma_comp_trait = self._scaled_param(self._params['sigma_comp_trait'], dt)
                sigma_comp_dist = self._scaled_param(self._params['sigma_comp_dist'], dt)
            else:
                sigma_env_trait = self._params['sigma_env_trait']
                sigma_comp_trait = self._params['sigma_comp_trait']
                sigma_comp_dist = self._params['sigma_comp_dist']

            x = self._individuals['x']/self._grid_bounds['x'][1]
            y = self._individuals['y']/self._grid_bounds['y'][1]
            # trait distance among individuals
            delta_comp_trait = dist.squareform(dist.pdist(self._individuals['trait']))
            delta_trait_norm = np.exp(-0.5 * delta_comp_trait ** 2 / sigma_comp_trait ** 2)
            # spatial distance among individuals
            delta_xy = dist.squareform(dist.pdist(np.column_stack([x, y])))
            delta_xy_norm = np.exp(-0.5 * delta_xy ** 2 / sigma_comp_dist ** 2)
            # number of individual with similar traits and in proximity to each other
            n_eff = 1 / (2 * np.pi * sigma_comp_dist ** 2) * np.sum(delta_trait_norm * delta_xy_norm, axis=1)
            delta_trait_i = self._individuals['trait'] - opt_trait
            diag_val = np.ones(self._individuals['trait'].shape[1]) * sigma_env_trait
            sigma_diag = np.diag(diag_val ** 2)
            sigma_offdiag = self._params['rho'] * (diag_val[:, np.newaxis] * diag_val[np.newaxis, :] - sigma_diag)
            sigma_i = sigma_diag + sigma_offdiag
            env_fitness = np.zeros(self._individuals['trait'].shape[0])
            inv_sigma = np.linalg.inv(sigma_i)
            for i in range(self._individuals['trait'].shape[0]):
                env_fitness[i] = np.exp(-1 / 2 * np.dot(delta_trait_i[i], np.dot(inv_sigma, delta_trait_i[i].T)))

            death_i = n_eff/(self._params['car_cap_max'] * env_fitness)
            birth_i = env_fitness * self._params['birth_rate'] #np.repeat(self._params['birth_rate'], self.abundance)
            #movement_i = np.repeat(self._params['movement_rate'], self.abundance)
            events_tot = np.sum(birth_i) + np.sum(death_i) #+ np.sum(movement_i)

            events_i = self._rng.choice(a=['B', 'D',  #'M'
                                           ], size=self._individuals['trait'].shape[0],
                                        p=[np.sum(birth_i) / events_tot, np.sum(death_i) / events_tot,
                                           #np.sum(movement_i) / events_tot
                                           ])

            tau = 0.0
            n_offspring = np.zeros(self._individuals['trait'].shape[0])
            while tau <= 1:
                dtau = self._rng.exponential(1 / events_tot)
                tau += dtau
                event_type_i = self._rng.choice(events_i)
                if event_type_i == 'B':
                    i_idx = self._rng.choice(a=self.abundance, p=birth_i/np.sum(birth_i))
                    n_offspring[i_idx] += self._params['car_cap_max']/n_eff[i_idx] * env_fitness[i_idx]
                    #pdb.set_trace()
                # elif event_type_i == 'D':
                #     i_idx = self._rng.choice(a=self.abundance, p=death_i / np.sum(death_i))
                #     n_offspring[i_idx] -= 1
                # elif event_type_i == 'M':
                #     i_idx = self._rng.choice(a=self.abundance, p=movement_i / np.sum(movement_i))
                #     new_x = self._truncnorm.rvs(a=(0-x[i_idx])/self._params['sigma_mov'],
                #                                 b=(1-x[i_idx])/self._params['sigma_mov'],
                #                                 loc=x[i_idx],
                #                                 scale=self._params['sigma_mov'])
                #     new_y = self._truncnorm.rvs(a=(0-y[i_idx])/self._params['sigma_mov'],
                #                                 b=(1-y[i_idx])/self._params['sigma_mov'],
                #                                 loc=y[i_idx],
                #                                 scale=self._params['sigma_mov'])
                #     self.individuals['x'][i_idx] = new_x * self._grid_bounds['x'][1]
                #     self.individuals['y'][i_idx] = new_y * self._grid_bounds['y'][1]
                n_offspring[n_offspring<0] = 0
        else:
            events_i = np.zeros(self._individuals['trait'].shape[1])
            death_i = np.zeros(self._individuals['trait'].shape[1])
            n_offspring = np.zeros(self._individuals['trait'].shape[1])

        self._individuals.update({'events_i': events_i,
                                  'death_i': death_i,
                                  'n_offspring': n_offspring.astype('int')
                                  })

    def _update_individuals(self, dt):
        """
        Update individuals' data (birth, death, and move).
        Parameters
        ----------
        dt : float
            Time step duration.
        """
        # rescale parameters
        if self._rescale_rates:
            mut_prob = self._scaled_param(self._params['mut_prob'], dt)
            sigma_mov = self._scaled_param(self._params['sigma_mov'], dt)
            sigma_mut = self._scaled_param(self._params['sigma_mut'], dt)
        else:
            mut_prob = self._params['mut_prob']
            sigma_mov = self._params['sigma_mov']
            sigma_mut = self._params['sigma_mut']

        n_offspring = self._individuals['n_offspring']
        if not n_offspring.sum():
            # population total extinction
            if self._params['on_extinction'] == 'raise':
                raise RuntimeError("no offspring generated. "
                                   "Model execution has stopped.")

            if self._params['on_extinction'] == 'warn':
                warnings.warn("no offspring generated. "
                              "Model execution continues with no population.",
                              RuntimeWarning)

            new_individuals = {k: np.array([])
                               for k in ('x', 'y', 'trait')}
            new_individuals['trait'] = np.expand_dims(new_individuals['trait'], 1)
        else:
            # generate offspring
            new_individuals = {k: np.repeat(self._individuals[k], n_offspring)
                               for k in ('x', 'y')}
            new_individuals['trait'] = np.repeat(self._individuals['trait'], n_offspring, axis=0)

            # mutate offspring
            to_mutate = self._rng.uniform(0, 1, new_individuals['trait'].shape[0]) < mut_prob
            for i in range(new_individuals['trait'].shape[1]):
                new_individuals['trait'][:, i] = np.where(to_mutate,
                                                          self._mutate_trait(new_individuals['trait'][:, i], sigma_mut),
                                                          new_individuals['trait'][:, i])

            # disperse offspring within grid bounds
            new_x, new_y = self._mov_within_bounds(new_individuals['x'], new_individuals['y'], sigma_mov)
            new_individuals['x'] = new_x
            new_individuals['y'] = new_y

        self._individuals['step'] += 1
        self._individuals['time'] += dt
        self._individuals.update(new_individuals)
        if not n_offspring.sum():
            taxon_id, ancestor_id = np.array([]), np.array([])
        else:
            taxon_id, ancestor_id = self._compute_taxon_ids()
        self._individuals.update({'taxon_id': taxon_id})
        self._individuals.update({'ancestor_id': ancestor_id})

        # reset offspring data
        self._individuals.update({
            'events_i': np.zeros(self._individuals['trait'].shape[0]),
            'death_i': np.zeros(self._individuals['trait'].shape[0]),
            'n_offspring': np.zeros(self._individuals['trait'].shape[0])
        })
