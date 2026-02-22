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
