"""!
@file src/NSGAII.py
@package Gnowee

@defgroup NSGAII NSGAII

@brief Implementation of NSGA-II operations for multi-objective optimization.

This module implements the core NSGA-II operations including non-dominated sorting
and crowding distance calculation.

@author OpenHands AI

@date 2024

@copyright GNU GPLv3.0+
"""

import numpy as np
from MOParent import MOParent

class NSGAII:
    """!
    @ingroup NSGAII
    Class implementing NSGA-II operations for multi-objective optimization.
    """
    
    @staticmethod
    def fast_non_dominated_sort(population):
        """!
        Perform fast non-dominated sorting of the population.
        
        @param population: <em> list of MOParent objects </em>
            The population to be sorted.
            
        @return \e list: List of fronts, where each front is a list of solutions.
        """
        fronts = [[]]  # Initialize first front
        
        for p in population:
            p.domination_count = 0  # Number of solutions dominating p
            p.dominated_solutions = []  # Solutions that p dominates
            
            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            
            if p.domination_count == 0:  # p belongs to first front
                p.rank = 1
                fronts[0].append(p)
        
        i = 0  # Initialize front counter
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:  # q belongs to next front
                        q.rank = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty front

    @staticmethod
    def calculate_crowding_distance(front):
        """!
        Calculate crowding distance for solutions in a front.
        
        @param front: <em> list of MOParent objects </em>
            List of solutions in the current front.
        """
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        num_objectives = len(front[0].fitness_values)
        
        for solution in front:
            solution.crowding_distance = 0
        
        for m in range(num_objectives):
            front.sort(key=lambda x: x.fitness_values[m])
            
            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate crowding distance for intermediate points
            f_max = front[-1].fitness_values[m]
            f_min = front[0].fitness_values[m]
            scale = f_max - f_min if f_max != f_min else 1.0
            
            for i in range(1, len(front)-1):
                front[i].crowding_distance += (
                    front[i+1].fitness_values[m] - front[i-1].fitness_values[m]
                ) / scale

    @staticmethod
    def crowding_operator(individual_a, individual_b):
        """!
        Compare two individuals using crowded comparison operator.
        
        @param individual_a: <em> MOParent object </em>
            First individual to compare.
        @param individual_b: <em> MOParent object </em>
            Second individual to compare.
            
        @return \e boolean: True if individual_a is better than individual_b.
        """
        if individual_a.rank < individual_b.rank:
            return True
        if individual_a.rank > individual_b.rank:
            return False
        return individual_a.crowding_distance > individual_b.crowding_distance

    @staticmethod
    def select_parents(population, tournament_size):
        """!
        Select parents using binary tournament selection with crowding operator.
        
        @param population: <em> list of MOParent objects </em>
            The population to select from.
        @param tournament_size: \e integer
            Size of the tournament.
            
        @return \e MOParent: Selected parent.
        """
        tournament = np.random.choice(population, tournament_size, replace=False)
        best = tournament[0]
        
        for candidate in tournament[1:]:
            if NSGAII.crowding_operator(candidate, best):
                best = candidate
        
        return best