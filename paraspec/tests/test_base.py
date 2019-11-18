import copy
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from paraspec import ParapatricSpeciationModel


@pytest.fixture(scope='session')
def params():
    return {
        'nb_radius': 5,
        'lifespan': 1,
        'capacity': 5,
        'sigma_w': 0.5,
        'sigma_d': 4,
        'sigma_mut': 0.5,
        'm_freq': 0.04,
        'random_seed': 1234,
        'on_extinction': 'ignore'
    }


@pytest.fixture(scope='session')
def grid():
    X, Y = np.meshgrid(np.linspace(0, 20, 10), np.linspace(0, 10, 20))
    return X, Y


@pytest.fixture(scope='session')
def env_field(grid):
    return np.random.uniform(0, 1, grid[0].shape)


@pytest.fixture
def model(params, grid):
    X, Y = grid
    return ParapatricSpeciationModel(X, Y, 10, **params)


@pytest.fixture
def initialized_model(model, env_field):
    m = copy.deepcopy(model)
    m.initialize_population([env_field.min(), env_field.max()])
    return m


@pytest.fixture(scope='session')
def model_repr():
    return dedent("""\
    <ParapatricSpeciationModel (population: not initialized)>
    Parameters:
        nb_radius: 5
        lifespan: 1
        capacity: 5
        sigma_w: 0.5
        sigma_d: 4
        sigma_mut: 0.5
        m_freq: 0.04
        random_seed: 1234
        on_extinction: ignore
    """)


@pytest.fixture(scope='session')
def initialized_model_repr():
    return dedent("""\
    <ParapatricSpeciationModel (population: 10)>
    Parameters:
        nb_radius: 5
        lifespan: 1
        capacity: 5
        sigma_w: 0.5
        sigma_d: 4
        sigma_mut: 0.5
        m_freq: 0.04
        random_seed: 1234
        on_extinction: ignore
    """)


def _in_bounds(grid_coord, pop_coord):
    return (pop_coord.min() >= grid_coord.min()
            and pop_coord.max() <= grid_coord.max())


class TestParapatricSpeciationModel(object):

    def test_constructor(self):
        with pytest.raises(KeyError, match="not valid model parameters"):
            ParapatricSpeciationModel([0, 1, 2], [0, 1, 2], 10,
                                      invalid_param=0, invlaid_param2='1')

        with pytest.raises(ValueError, match="invalid value"):
            ParapatricSpeciationModel([0, 1, 2], [0, 1, 2], 10,
                                      on_extinction='invalid')

        rs = np.random.RandomState(0)

        m = ParapatricSpeciationModel([0, 1, 2], [0, 1, 2], 10, random_seed=rs)
        assert m._random is rs

        m2 = ParapatricSpeciationModel([0, 1, 2], [0, 1, 2], 10, random_seed=0)
        np.testing.assert_equal(m2._random.get_state()[1], rs.get_state()[1])

    def test_params(self, params, model):
        assert model.params == params

    def test_initialize_population(self, grid, initialized_model):
        assert initialized_model.population_size == 10

        assert initialized_model.population['step'] == 0
        np.testing.assert_equal(initialized_model.population['id'],
                                np.arange(0, 10))
        np.testing.assert_equal(initialized_model.population['parent'],
                                np.arange(0, 10))

        trait = initialized_model.population['trait']
        assert np.all((trait >= 0) & (trait <= 1))

        assert _in_bounds(grid[0], initialized_model.population['x'])
        assert _in_bounds(grid[1], initialized_model.population['y'])

    @pytest.mark.parametrize("x_range,y_range,error", [
        (None, None, False),
        ([0, 15], None, False),
        (None, [2, 7], False),
        ([0, 15], [2, 7], False),
        ([-1, 100], None, True),
        (None, [-1, 100], True),
        ([-1, 100], [-1, 100], True)
    ])
    def test_xy_range(self, model, grid, x_range, y_range, error):
        if error:
            expected = "x_range and y_range must be within model bounds"
            with pytest.raises(ValueError, match=expected):
                model.initialize_population(
                    [0, 1], x_range=x_range, y_range=y_range
                )

        else:
            model.initialize_population(
                [0, 1], x_range=x_range, y_range=y_range
            )
            x_r = x_range or grid[0]
            y_r = y_range or grid[1]
            assert _in_bounds(np.array(x_r), model.population['x'])
            assert _in_bounds(np.array(y_r), model.population['y'])

    def test_to_dataframe(self, initialized_model):
        expected = pd.DataFrame(initialized_model.population)
        actual = initialized_model.to_dataframe()
        pd.testing.assert_frame_equal(actual, expected)

    def test_scaled_params(self, model):
        params = model._get_scaled_params(4)
        expected = (0.5, 8., 1)

        assert params == expected

    def test_count_neighbors(self, model, grid):
        points = np.column_stack([[0, 4, 8, 12], [0, 2, 4, 6]])
        expected = [2, 3, 3, 2]

        np.testing.assert_equal(model._count_neighbors(points), expected)

    def test_get_optimal_trait(self, model, grid, env_field):
        # using points = grid points + offset less than grid spacing
        # expected: env_field and optimal trait should be equal
        X, Y = grid
        points = np.column_stack([X.ravel() + 0.1, Y.ravel() + 0.1])

        opt_trait = model._get_optimal_trait(env_field, points)

        np.testing.assert_array_equal(opt_trait, env_field.ravel())

    def test_update_population(self, model, grid, env_field):
        # do many runs to avoid favorable random conditions
        trait_diff = []

        for i in range(1000):
            model._random = np.random.RandomState(i)

            model.initialize_population([env_field.min(), env_field.max()])
            init_pop = model.population.copy()
            model.update_population(env_field, 1)
            current_pop = model.population.copy()

            # test step
            assert current_pop['step'] == 1
            assert current_pop['id'][0] == init_pop['id'].size

            # test dispersal (only check within domain)
            assert _in_bounds(grid[0], current_pop['x'])
            assert _in_bounds(grid[1], current_pop['y'])

            # test mutation
            model.update_population(env_field, 1)
            last_pop = model.population.copy()
            idx = np.searchsorted(current_pop['id'], last_pop['parent'])
            trait_diff.append(current_pop['trait'][idx] - last_pop['trait'])

        trait_diff = np.concatenate(trait_diff)
        trait_rms = np.sqrt(np.mean(trait_diff**2))
        scaled_sigma_mut = 1   # sigma_mut * sqrt(m_freq) * 1
        assert pytest.approx(trait_rms, scaled_sigma_mut)

    @pytest.mark.parametrize("nfreq", [None, 1, 10])
    def test_updade_population_nfreq(self, model, env_field, nfreq):
        model.initialize_population([env_field.min(), env_field.max()])
        model.update_population(env_field, 1, nfreq=nfreq)
        pop1 = model.population.copy()

        # test nfreq
        model.update_population(env_field, 1, nfreq=nfreq)
        pop2 = model.population.copy()
        step = pop1['step']

        old_parent = np.max(pop1['parent'])
        new_parent = np.max(pop2['parent'])

        if nfreq is None or not step % nfreq:
            assert old_parent < new_parent
        else:
            assert old_parent >= new_parent

    @pytest.mark.parametrize('capacity_mul,env_field_mul,on_extinction', [
        (0., 1, 'raise'),
        (0., 1, 'warn'),
        (0., 1, 'ignore'),
        (1., 1e3, 'ignore')
    ])
    def test_update_population_extinction(self,
                                          initialized_model,
                                          env_field,
                                          capacity_mul,
                                          env_field_mul,
                                          on_extinction):

        subset_keys = ('id', 'parent', 'x', 'y', 'trait')

        def get_pop_subset():
            pop = initialized_model.population.copy()
            return {k: pop[k] for k in subset_keys}

        initialized_model._params['on_extinction'] = on_extinction

        # no offspring via either r_d values = 0 or very low fitness values
        initialized_model._params['capacity'] *= capacity_mul
        field = env_field * env_field_mul

        if on_extinction == 'raise':
            with pytest.raises(RuntimeError, match="no offspring"):
                initialized_model.update_population(field, 1)
            return

        elif on_extinction == 'warn':
            with pytest.warns(RuntimeWarning, match="no offspring"):
                initialized_model.update_population(field, 1)
                current = get_pop_subset()
                initialized_model.update_population(field, 1)
                next = get_pop_subset()

        else:
            initialized_model.update_population(field, 1)
            current = get_pop_subset()
            initialized_model.update_population(field, 1)
            next = get_pop_subset()

        for k in subset_keys:
            assert current[k].size == 0
            assert next[k].size == 0

    def test_repr(self, model, model_repr,
                  initialized_model, initialized_model_repr):
        assert repr(model) == model_repr
        assert repr(initialized_model) == initialized_model_repr
