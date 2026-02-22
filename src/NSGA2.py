"""!
@file src/NSGA2.py
@package Gnowee

@defgroup NSGA2 NSGA2

@brief Non-dominated Sorting Genetic Algorithm II (NSGA-II) for
multi-objective optimization.

This module implements the NSGA-II algorithm as described in:

    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II.
    IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

Key components:
  - fast_non_dominated_sort: assign Pareto front ranks to a population
  - crowding_distance: diversity-preserving secondary sorting criterion
  - NSGA2: main class encapsulating the full algorithm
  - nsga2_main: convenience entry point

Multi-objective benchmark functions (ZDT1, ZDT2, ZDT3, Schaffer N1,
Kursawe) are available via ObjectiveFunction.

@author Gnowee contributors

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

import copy

import numpy as np
from numpy.random import rand

# ---------------------------------------------------------------------------
# Pareto-front utilities
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(objectives):
    """!
    @ingroup NSGA2
    Assign each individual a Pareto-front rank using Deb's fast
    non-dominated sorting procedure.

    Complexity: O(M * N^2) where M = number of objectives, N = population
    size.

    @param objectives: <em> list of lists or 2-D array </em> \n
        objectives[i] is the vector of objective values for individual i.
        Lower values are considered better (minimisation). \n

    @return \\e list of lists: fronts[k] contains the indices of individuals
        that belong to Pareto front k (0 = best). \n
    @return \\e list: rank[i] is the front index assigned to individual i. \n
    """
    n = len(objectives)
    dom_count = [0] * n          # number of solutions that dominate i
    dom_set = [[] for _ in range(n)]  # solutions dominated by i
    fronts = [[]]
    rank = [0] * n

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dom_set[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dom_set[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove the empty sentinel appended in the last iteration
    if not fronts[-1]:
        fronts.pop()

    return fronts, rank


def crowding_distance(objectives, front):
    """!
    @ingroup NSGA2
    Compute the crowding distance for each individual in a single front.

    Boundary solutions (lowest and highest objective values per objective)
    are assigned infinite crowding distance so they are always preserved.

    @param objectives: <em> list of lists or 2-D array </em> \n
        Full population objective matrix (all fronts). \n
    @param front: \\e list \n
        Indices of individuals belonging to the front of interest. \n

    @return \\e dict: Mapping from individual index to crowding distance. \n
    """
    distances = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distances[idx] = float('inf')
        return distances

    num_obj = len(objectives[0])
    for m in range(num_obj):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        obj_min = objectives[sorted_front[0]][m]
        obj_max = objectives[sorted_front[-1]][m]
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        span = obj_max - obj_min
        if span == 0.0:
            continue
        for k in range(1, len(sorted_front) - 1):
            distances[sorted_front[k]] += (
                objectives[sorted_front[k + 1]][m]
                - objectives[sorted_front[k - 1]][m]
            ) / span

    return distances


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dominates(obj_a, obj_b):
    """Return True if obj_a dominates obj_b (all <=, at least one <)."""
    at_least_one_better = False
    for a, b in zip(obj_a, obj_b):
        if a > b:
            return False
        if a < b:
            at_least_one_better = True
    return at_least_one_better


def _sbx_crossover(parent1, parent2, lb, ub, eta_c=20.0):
    """
    Simulated Binary Crossover (SBX).

    @param parent1, parent2: \\e array \n
        Parent decision vectors. \n
    @param lb, ub: \\e array \n
        Lower and upper bounds. \n
    @param eta_c: \\e float \n
        Distribution index (higher = offspring closer to parents). \n

    @return \\e tuple: Two child decision vectors. \n
    """
    child1 = copy.copy(parent1)
    child2 = copy.copy(parent2)
    n = len(parent1)
    for i in range(n):
        if rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                y1 = min(parent1[i], parent2[i])
                y2 = max(parent1[i], parent2[i])
                yl = lb[i]
                yu = ub[i]
                rand_u = rand()
                beta = 1.0 + 2.0 * (y1 - yl) / (y2 - y1)
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                betaq = _beta_q(rand_u, alpha, eta_c)
                child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                beta = 1.0 + 2.0 * (yu - y2) / (y2 - y1)
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                betaq = _beta_q(rand_u, alpha, eta_c)
                child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                child1[i] = np.clip(child1[i], yl, yu)
                child2[i] = np.clip(child2[i], yl, yu)

                if rand() > 0.5:
                    child1[i], child2[i] = child2[i], child1[i]

    return child1, child2


def _beta_q(u, alpha, eta):
    """Helper for SBX spread factor."""
    if u <= 1.0 / alpha:
        return (u * alpha) ** (1.0 / (eta + 1.0))
    return (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))


def _polynomial_mutation(individual, lb, ub, eta_m=20.0, prob=None):
    """
    Polynomial mutation operator.

    @param individual: \\e array \n
        Decision vector to mutate (modified in-place). \n
    @param lb, ub: \\e array \n
        Lower and upper bounds. \n
    @param eta_m: \\e float \n
        Distribution index. \n
    @param prob: \\e float \n
        Per-variable mutation probability (default: 1/n). \n

    @return \\e array: Mutated decision vector. \n
    """
    n = len(individual)
    if prob is None:
        prob = 1.0 / n
    child = copy.copy(individual)
    for i in range(n):
        if rand() < prob:
            yl = lb[i]
            yu = ub[i]
            delta1 = (child[i] - yl) / (yu - yl)
            delta2 = (yu - child[i]) / (yu - yl)
            u = rand()
            mut_pow = 1.0 / (eta_m + 1.0)
            if u <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta_m + 1.0)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta_m + 1.0)
                delta_q = 1.0 - val ** mut_pow
            child[i] = np.clip(child[i] + delta_q * (yu - yl), yl, yu)
    return child


def _tournament_select(population, objectives, rank, distances, k=2):
    """
    Binary tournament selection using crowded comparison operator.

    Selects the better of k randomly drawn candidates using the
    crowded-comparison criterion: lower rank wins; ties broken by
    larger crowding distance.

    @return \\e array: Selected individual's decision vector. \n
    """
    candidates = np.random.choice(len(population), k, replace=False)
    best = candidates[0]
    for c in candidates[1:]:
        if rank[c] < rank[best]:
            best = c
        elif rank[c] == rank[best] and distances.get(c, 0) > distances.get(best, 0):
            best = c
    return copy.copy(population[best])


# ---------------------------------------------------------------------------
# NSGA-II main class
# ---------------------------------------------------------------------------

class NSGA2:
    """!
    @ingroup NSGA2
    Non-dominated Sorting Genetic Algorithm II (NSGA-II).

    Supports continuous variable problems.  Mixed-integer extensions can be
    added by rounding integer variables after crossover/mutation.

    Typical usage::

        from NSGA2 import NSGA2
        from ObjectiveFunction import ObjectiveFunction

        of = ObjectiveFunction(method='zdt1')
        optimizer = NSGA2(
            obj_func=of.func,
            lb=[0.0] * 30,
            ub=[1.0] * 30,
            population=100,
            max_gen=250,
        )
        result = optimizer.run()
        # result['pareto_front'] – list of non-dominated objective vectors
        # result['pareto_set']   – corresponding decision vectors
    """

    def __init__(self, obj_func, lb, ub, population=100, max_gen=250,
                 eta_c=20.0, eta_m=20.0, crossover_prob=0.9,
                 mutation_prob=None, seed=None):
        """!
        Constructor.

        @param obj_func: <em> callable </em> \n
            Multi-objective function returning a list/array of objective
            values. Lower is better for every objective. \n
        @param lb: \\e array-like \n
            Lower bounds on decision variables. \n
        @param ub: \\e array-like \n
            Upper bounds on decision variables. \n
        @param population: \\e integer \n
            Population size N (must be even). \n
        @param max_gen: \\e integer \n
            Maximum number of generations. \n
        @param eta_c: \\e float \n
            SBX crossover distribution index. \n
        @param eta_m: \\e float \n
            Polynomial mutation distribution index. \n
        @param crossover_prob: \\e float \n
            Probability of applying crossover to a pair. \n
        @param mutation_prob: \\e float or None \n
            Per-variable mutation probability (default: 1/n_vars). \n
        @param seed: \\e integer or None \n
            Random seed for reproducibility. \n
        """
        if seed is not None:
            np.random.seed(seed)

        self.obj_func = obj_func
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.n_vars = len(lb)
        self.population = population + (population % 2)  # ensure even
        self.max_gen = max_gen
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob if mutation_prob is not None \
            else 1.0 / self.n_vars

    # ------------------------------------------------------------------
    def run(self):
        """!
        Execute the NSGA-II optimisation loop.

        @return \\e dict with keys:

          - ``'pareto_front'``: list of objective vectors on the
            final Pareto-approximation front (front 0).
          - ``'pareto_set'``: corresponding decision vectors.
          - ``'all_objectives'``: objective vectors for the full final
            population.
          - ``'all_variables'``: decision vectors for the full final
            population.
          - ``'generations'``: number of generations run.
        """
        # --- initialise population ---
        pop = [self.lb + rand(self.n_vars) * (self.ub - self.lb)
               for _ in range(self.population)]
        obj = [list(self.obj_func(x)) for x in pop]

        for gen in range(self.max_gen):
            # Non-dominated sort + crowding distance for parent pool
            fronts, rank = fast_non_dominated_sort(obj)
            distances = {}
            for front in fronts:
                distances.update(crowding_distance(obj, front))

            # --- generate offspring ---
            offspring = []
            while len(offspring) < self.population:
                p1 = _tournament_select(pop, obj, rank, distances)
                p2 = _tournament_select(pop, obj, rank, distances)
                if rand() < self.crossover_prob:
                    c1, c2 = _sbx_crossover(p1, p2, self.lb, self.ub,
                                            self.eta_c)
                else:
                    c1, c2 = copy.copy(p1), copy.copy(p2)
                c1 = _polynomial_mutation(c1, self.lb, self.ub, self.eta_m,
                                          self.mutation_prob)
                c2 = _polynomial_mutation(c2, self.lb, self.ub, self.eta_m,
                                          self.mutation_prob)
                offspring.extend([c1, c2])

            offspring = offspring[:self.population]
            off_obj = [list(self.obj_func(x)) for x in offspring]

            # --- combine parent + offspring ---
            combined_pop = pop + offspring
            combined_obj = obj + off_obj

            # --- select next generation by non-dominated sorting ---
            fronts, rank = fast_non_dominated_sort(combined_obj)
            distances = {}
            for front in fronts:
                distances.update(crowding_distance(combined_obj, front))

            new_pop = []
            new_obj = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.population:
                    for idx in front:
                        new_pop.append(copy.copy(combined_pop[idx]))
                        new_obj.append(combined_obj[idx])
                else:
                    # Fill remainder sorted by crowding distance
                    remaining = self.population - len(new_pop)
                    sorted_front = sorted(
                        front,
                        key=lambda i: distances.get(i, 0),
                        reverse=True,
                    )
                    for idx in sorted_front[:remaining]:
                        new_pop.append(copy.copy(combined_pop[idx]))
                        new_obj.append(combined_obj[idx])
                    break

            pop = new_pop
            obj = new_obj

        # --- extract final Pareto front ---
        fronts, _ = fast_non_dominated_sort(obj)
        pareto_indices = fronts[0]
        return {
            'pareto_front': [obj[i] for i in pareto_indices],
            'pareto_set': [pop[i] for i in pareto_indices],
            'all_objectives': obj,
            'all_variables': pop,
            'generations': self.max_gen,
        }


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def nsga2_main(obj_func, lb, ub, population=100, max_gen=250, **kwargs):
    """!
    @ingroup NSGA2
    Convenience wrapper that creates an NSGA2 instance and calls run().

    @param obj_func: <em> callable </em> \n
        Multi-objective function returning a list of objective values. \n
    @param lb: \\e array-like \n
        Lower bounds. \n
    @param ub: \\e array-like \n
        Upper bounds. \n
    @param population: \\e integer \n
        Population size. \n
    @param max_gen: \\e integer \n
        Maximum generations. \n
    @param kwargs: Additional keyword arguments forwarded to NSGA2. \n

    @return \\e dict: Optimisation result (see NSGA2.run()). \n
    """
    optimizer = NSGA2(obj_func, lb, ub, population=population,
                      max_gen=max_gen, **kwargs)
    return optimizer.run()
