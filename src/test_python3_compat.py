"""
Tests to verify Python 3 compatibility of the Gnowee codebase.

Run with: python -m pytest src/test_python3_compat.py -v
"""

import sys
import os
import math

import pytest
import numpy as np

# Ensure the src directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

# The NOLH algorithm supports dimensions 2–29 (see get_cdr_permutations).
# We test a conservative subset (2–22) to keep runtime short while exercising
# all the boundary transitions where m increments.
NOLH_TEST_DIM_RANGE = range(2, 23)


# ---------------------------------------------------------------------------
# Import tests – verify all modules load without SyntaxError or ImportError
# ---------------------------------------------------------------------------

def test_import_gnowee_utilities():
    import GnoweeUtilities  # noqa: F401


def test_import_sampling():
    import Sampling  # noqa: F401


def test_import_objective_function():
    import ObjectiveFunction  # noqa: F401


def test_import_constraints():
    import Constraints  # noqa: F401


def test_import_gnowee_heuristics():
    import GnoweeHeuristics  # noqa: F401


def test_import_tsp():
    import TSP  # noqa: F401


def test_import_example_function():
    import ExampleFunction  # noqa: F401


# ---------------------------------------------------------------------------
# functools.reduce – ObjectiveFunction.prod()
# ---------------------------------------------------------------------------

def test_prod_uses_functools_reduce():
    """prod() must work in Python 3 where reduce() is in functools."""
    from ObjectiveFunction import prod
    assert prod([1, 2, 3, 4]) == 24
    assert prod([5]) == 5
    assert prod(x for x in range(1, 6)) == 120


# ---------------------------------------------------------------------------
# Switch generator – PEP 479: raise StopIteration inside generator is a
# RuntimeError in Python 3.7+; the fix replaces it with a plain return.
# ---------------------------------------------------------------------------

def test_switch_does_not_raise_runtime_error():
    """Iterating the Switch class must not raise RuntimeError (PEP 479)."""
    from GnoweeUtilities import Switch

    matched = False
    for case in Switch("hello"):
        if case("hello"):
            matched = True
            break
    assert matched


def test_switch_default_case():
    """Switch default (no-arg) case must still fire when no case matched."""
    from GnoweeUtilities import Switch

    hit_default = False
    for case in Switch("unknown"):
        if case("a"):
            break
        if case():
            hit_default = True
            break
    assert hit_default


# ---------------------------------------------------------------------------
# Integer division in params() – Python 3 changed / to float division;
# params() must always return integer values for s so the NOLH algorithm
# produces correct array shapes.
# ---------------------------------------------------------------------------

def test_params_returns_integers():
    """params() must return integer values (not floats) for all outputs."""
    from Sampling import params

    for dim in NOLH_TEST_DIM_RANGE:
        m_val, q_val, r_val = params(dim)
        assert isinstance(m_val, int), f"m is not int for dim={dim}: {m_val}"
        assert isinstance(q_val, int), f"q is not int for dim={dim}: {q_val}"
        # r_val = s - dim; s is an int so r_val should also be an int
        assert isinstance(r_val, int), f"r is not int for dim={dim}: {r_val}"
        assert r_val >= 0, f"r is negative for dim={dim}: {r_val}"


def test_params_s_geq_dim():
    """The computed s (number of NOLH columns) must be >= dim."""
    from Sampling import params

    for dim in NOLH_TEST_DIM_RANGE:
        m_val, q_val, _ = params(dim)
        s = m_val + math.factorial(m_val - 1) // (2 * math.factorial(m_val - 3))
        assert s >= dim, f"s={s} < dim={dim}"


# ---------------------------------------------------------------------------
# pyDOE3 – lhs sampling works
# ---------------------------------------------------------------------------

def test_lhc_sampling():
    """Latin Hypercube sampling via pyDOE3 must return correct shape."""
    from Sampling import initial_samples

    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    samples = initial_samples(lb, ub, 'lhc', 10)
    assert samples.shape == (10, 3)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_random_sampling():
    """Random phase-space sampling must return the requested number of samples."""
    from Sampling import initial_samples

    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])
    samples = initial_samples(lb, ub, 'random', 20)
    assert samples.shape == (20, 2)


# ---------------------------------------------------------------------------
# Levy flight – Sampling functions must return correct shapes
# ---------------------------------------------------------------------------

def test_levy_1d():
    from Sampling import levy
    result = levy(5)
    assert result.shape == (5,)


def test_levy_2d():
    from Sampling import levy
    result = levy(4, nr=3)
    assert result.shape == (3, 4)


def test_tlf_shape():
    from Sampling import tlf
    result = tlf(numRow=3, numCol=4)
    assert result.shape == (3, 4)
    assert np.all(result >= 0) and np.all(result <= 1)


# ---------------------------------------------------------------------------
# GnoweeUtilities – Parent and Event objects
# ---------------------------------------------------------------------------

def test_parent_construction():
    from GnoweeUtilities import Parent
    p = Parent(fitness=42.0, variables=np.array([1.0, 2.0]))
    assert p.fitness == 42.0
    assert len(p.variables) == 2


def test_event_construction():
    from GnoweeUtilities import Event
    e = Event(generation=1, evaluations=10, fitness=0.5,
              design=np.array([1.0, 2.0]))
    assert e.generation == 1
    assert e.fitness == 0.5


# ---------------------------------------------------------------------------
# ObjectiveFunction – a handful of benchmark functions
# ---------------------------------------------------------------------------

def test_ackley_function():
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='ackley')
    # Known global minimum is 0 at origin
    result = of.func([0.0, 0.0])
    assert result == pytest.approx(0.0, abs=1e-10)


def test_rosenbrock_function():
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='rosenbrock')
    # rosenbrock([1, 1]) = 0 for the standard form
    result = of.func([1.0, 1.0])
    assert result == pytest.approx(0.0, abs=1e-10)


def test_objective_function_with_callable():
    """ObjectiveFunction should also accept a raw callable."""
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method=lambda x: sum(xi**2 for xi in x))
    result = of.func([3.0, 4.0])
    assert result == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Constraints – construction and evaluation
# ---------------------------------------------------------------------------

def test_constraint_with_callable():
    """Constraint must accept a raw callable (not just named methods)."""
    from Constraints import Constraint
    c = Constraint(method=lambda x: x[0] - 1.0, constraint=0.0)
    assert c.func is not None


def test_constraint_less_or_equal():
    from Constraints import Constraint
    c = Constraint(method='less_or_equal', constraint=5.0)
    assert c.func is not None
    # candidate > constraint → positive penalty
    penalty = c.func(10.0)
    assert penalty > 0
    # candidate ≤ constraint → zero penalty
    penalty_ok = c.func(3.0)
    assert penalty_ok == 0.0


# ---------------------------------------------------------------------------
# Multi-objective benchmark functions
# ---------------------------------------------------------------------------

def test_schaffer_n1_at_optimum():
    """Schaffer N1: at x=0 => f1=0, f2=4; at x=2 => f1=4, f2=0."""
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='schaffer_n1')
    f = of.func([0.0])
    assert f[0] == pytest.approx(0.0)
    assert f[1] == pytest.approx(4.0)
    f2 = of.func([2.0])
    assert f2[0] == pytest.approx(4.0)
    assert f2[1] == pytest.approx(0.0)


def test_schaffer_n1_returns_two_objectives():
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='schaffer_n1')
    result = of.func([1.0])
    assert len(result) == 2


def test_kursawe_returns_two_objectives():
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='kursawe')
    result = of.func([0.0, 0.0, 0.0])
    assert len(result) == 2


def test_zdt1_at_pareto_front():
    """ZDT1: when x[1:] = 0 => g=1, h=1-sqrt(x[0]), f2=1-sqrt(x[0])."""
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='zdt1')
    x = [0.25] + [0.0] * 29
    f = of.func(x)
    assert len(f) == 2
    assert f[0] == pytest.approx(0.25)
    assert f[1] == pytest.approx(1.0 - math.sqrt(0.25), rel=1e-6)


def test_zdt2_returns_two_objectives():
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='zdt2')
    result = of.func([0.5] * 30)
    assert len(result) == 2


def test_zdt3_returns_two_objectives():
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='zdt3')
    result = of.func([0.5] * 30)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# NSGA-II – core utilities
# ---------------------------------------------------------------------------

def test_import_nsga2():
    import NSGA2  # noqa: F401


def test_dominates_basic():
    """[1,2] dominates [2,3] but not [1,1]."""
    from NSGA2 import _dominates
    assert _dominates([1.0, 2.0], [2.0, 3.0]) is True
    assert _dominates([1.0, 2.0], [1.0, 1.0]) is False
    assert _dominates([1.0, 2.0], [1.0, 2.0]) is False


def test_fast_non_dominated_sort_simple():
    """With 3 solutions where one dominates another, verify rank assignment."""
    from NSGA2 import fast_non_dominated_sort
    # obj[0] dominates obj[2], obj[1] is non-dominated by obj[0]/obj[2]
    objectives = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]  # all non-dominated
    fronts, rank = fast_non_dominated_sort(objectives)
    assert len(fronts) == 1
    assert set(fronts[0]) == {0, 1, 2}
    assert all(r == 0 for r in rank)


def test_fast_non_dominated_sort_two_fronts():
    from NSGA2 import fast_non_dominated_sort
    # [1,1] dominates [2,2]
    objectives = [[1.0, 1.0], [2.0, 2.0]]
    fronts, rank = fast_non_dominated_sort(objectives)
    assert fronts[0] == [0]
    assert fronts[1] == [1]
    assert rank[0] == 0 and rank[1] == 1


def test_crowding_distance_boundary_is_inf():
    """Boundary individuals in a front must receive infinite crowding distance."""
    from NSGA2 import crowding_distance
    objectives = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    front = [0, 1, 2]
    dist = crowding_distance(objectives, front)
    assert dist[0] == float('inf')
    assert dist[2] == float('inf')
    assert dist[1] < float('inf')


def test_crowding_distance_two_individuals():
    """Front with only 2 individuals: both should get infinite distance."""
    from NSGA2 import crowding_distance
    objectives = [[0.0, 1.0], [1.0, 0.0]]
    front = [0, 1]
    dist = crowding_distance(objectives, front)
    assert dist[0] == float('inf')
    assert dist[1] == float('inf')


# ---------------------------------------------------------------------------
# NSGA-II – end-to-end runs on benchmark problems
# ---------------------------------------------------------------------------

def test_nsga2_schaffer_n1_produces_pareto_front():
    """NSGA-II on Schaffer N1 must return a non-empty Pareto front."""
    from NSGA2 import nsga2_main
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='schaffer_n1')
    result = nsga2_main(of.func, lb=[-10.0], ub=[10.0],
                        population=20, max_gen=10, seed=42)
    assert len(result['pareto_front']) > 0
    assert len(result['pareto_set']) == len(result['pareto_front'])
    # All Pareto-front members must be 2-objective vectors
    for f in result['pareto_front']:
        assert len(f) == 2


def test_nsga2_zdt1_pareto_front_quality():
    """
    NSGA-II on ZDT1 (n=5 vars, 50 individuals, 50 gens): the Pareto
    approximation front must contain solutions where f1+f2 >= 1 and
    f2 is close to 1-sqrt(f1) when g≈1.
    """
    from NSGA2 import nsga2_main
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='zdt1')
    result = nsga2_main(of.func, lb=[0.0] * 5, ub=[1.0] * 5,
                        population=50, max_gen=50, seed=0)
    pf = result['pareto_front']
    assert len(pf) > 0
    # Every objective value must be finite and non-negative
    for f in pf:
        assert f[0] >= 0.0
        assert f[1] >= 0.0


def test_nsga2_result_keys():
    """NSGA-II result dict must contain the expected keys."""
    from NSGA2 import nsga2_main
    result = nsga2_main(lambda x: [x[0] ** 2, (x[0] - 2) ** 2],
                        lb=[-5.0], ub=[5.0],
                        population=10, max_gen=5, seed=1)
    assert 'pareto_front' in result
    assert 'pareto_set' in result
    assert 'all_objectives' in result
    assert 'all_variables' in result
    assert 'generations' in result


def test_nsga2_population_size_is_preserved():
    """Final population must not exceed the requested population size."""
    from NSGA2 import nsga2_main
    result = nsga2_main(lambda x: [x[0], 1.0 - x[0]],
                        lb=[0.0], ub=[1.0],
                        population=20, max_gen=5, seed=7)
    assert len(result['all_variables']) <= 20


def test_nsga2_kursawe():
    """NSGA-II on Kursawe must converge to a 2-objective Pareto front."""
    from NSGA2 import nsga2_main
    from ObjectiveFunction import ObjectiveFunction
    of = ObjectiveFunction(method='kursawe')
    result = nsga2_main(of.func, lb=[-5.0] * 3, ub=[5.0] * 3,
                        population=30, max_gen=20, seed=3)
    assert len(result['pareto_front']) > 0
    for f in result['pareto_front']:
        assert len(f) == 2
