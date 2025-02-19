"""!
@file src/MOGnowee.py
@package Gnowee

@defgroup MOGnowee MOGnowee

@brief Multi-objective extension of Gnowee using NSGA-II.

This module extends Gnowee to support multi-objective optimization using NSGA-II
algorithm features while maintaining Gnowee's core heuristics.

@author OpenHands AI

@date 2024

@copyright GNU GPLv3.0+
"""

import numpy as np
from GnoweeHeuristics import GnoweeHeuristics
from MOParent import MOParent
from NSGAII import NSGAII
from MOObjectiveFunction import MOObjectiveFunction

class MOGnowee(GnoweeHeuristics):
    """!
    @ingroup MOGnowee
    Class implementing multi-objective optimization using NSGA-II with Gnowee's heuristics.
    """

    def __init__(self, *args, **kwargs):
        """!
        Constructor to build the MOGnowee class.
        
        Inherits all parameters from GnoweeHeuristics and adds multi-objective
        specific parameters.
        """
        super().__init__(*args, **kwargs)
        
        # Ensure objective is MOObjectiveFunction
        if not isinstance(self.objective, MOObjectiveFunction):
            raise TypeError("Objective must be an MOObjectiveFunction instance")

    def population_update(self, pop, children, timeline=None, adoptedParents=None,
                        mhFrac=0.0, randomParents=False):
        """!
        Update population using NSGA-II selection.
        
        @param self: <em> MOGnowee pointer </em>
            The MOGnowee pointer.
        @param pop: <em> list of MOParent objects </em>
            Current population.
        @param children: <em> list of arrays </em>
            List of child solutions to evaluate.
        @param timeline: <em> list of Event objects </em>
            List tracking the optimization progress.
        @param adoptedParents: \e array
            Indices of parents that were used to create children.
        @param mhFrac: \e float
            Fraction of population to replace with Metropolis criteria.
        @param randomParents: \e boolean
            Flag indicating if parents should be selected randomly.
            
        @return \e tuple: Updated population, number of changes, and timeline.
        """
        # Create MOParent objects for children
        child_pop = []
        for child in children:
            # Evaluate all objectives for child
            fitness_values = self.objective.evaluate(child)
            child_pop.append(MOParent(variables=child, fitness=fitness_values))

        # Combine parents and children
        combined_pop = pop + child_pop

        # Perform non-dominated sorting
        fronts = NSGAII.fast_non_dominated_sort(combined_pop)

        # Calculate crowding distance for each front
        for front in fronts:
            NSGAII.calculate_crowding_distance(front)

        # Create new population starting from the best front
        new_pop = []
        front_idx = 0
        while len(new_pop) + len(fronts[front_idx]) <= self.population:
            # Add whole front
            new_pop.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break

        # If needed, sort last front by crowding distance and add best solutions
        if len(new_pop) < self.population and front_idx < len(fronts):
            last_front = fronts[front_idx]
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            new_pop.extend(last_front[:self.population - len(new_pop)])

        # Update timeline with non-dominated solutions
        if timeline is not None:
            # Get the first front (non-dominated solutions)
            pareto_front = fronts[0]
            
            # Add each non-dominated solution to timeline
            for solution in pareto_front:
                timeline.append(self.create_event(solution))

        return new_pop, len(child_pop), timeline

    def create_event(self, solution):
        """!
        Create an event object for tracking optimization progress.
        
        @param self: <em> MOGnowee pointer </em>
            The MOGnowee pointer.
        @param solution: <em> MOParent object </em>
            The solution to create an event for.
            
        @return \e Event: Event object tracking the solution.
        """
        from GnoweeUtilities import Event
        return Event(
            generation=0,  # Will be updated in main loop
            evaluations=len(solution.fitness_values),
            fitness=solution.fitness_values,  # Store all objective values
            design=solution.variables
        )

    def select_parents(self, population, num_parents):
        """!
        Select parents using NSGA-II tournament selection.
        
        @param self: <em> MOGnowee pointer </em>
            The MOGnowee pointer.
        @param population: <em> list of MOParent objects </em>
            The population to select from.
        @param num_parents: \e integer
            Number of parents to select.
            
        @return \e list: Selected parents.
        """
        parents = []
        for _ in range(num_parents):
            parent = NSGAII.select_parents(population, tournament_size=2)
            parents.append(parent)
        return parents