"""!
@file src/MOParent.py
@package Gnowee

@defgroup MOParent MOParent

@brief Class for multi-objective optimization parent representation.

This class extends the Parent class to support multi-objective optimization
using NSGA-II algorithm features like non-dominated sorting and crowding distance.

@author OpenHands AI

@date 2024

@copyright GNU GPLv3.0+
"""

import numpy as np
from GnoweeUtilities import Parent

class MOParent(Parent):
    """!
    @ingroup MOParent
    The class extends Parent to support multi-objective optimization using NSGA-II.
    """

    def __init__(self, variables=None, fitness=None, changeCount=0, stallCount=0):
        """!
        Constructor to build the MOParent class.

        @param self: <em> MOParent pointer </em>
            The MOParent pointer.
        @param variables: \e array
            The set of variables representing a design solution.
        @param fitness: \e array
            The assessed fitness values for each objective.
        @param changeCount: \e integer
            The number of improvements to the current population member.
        @param stallCount: \e integer
            The number of evaluations since the last improvement.
        """
        super().__init__(variables, float('inf'), changeCount, stallCount)
        
        ## @var fitness_values
        # \e array:
        # The assessed fitness values for each objective.
        self.fitness_values = np.array([float('inf')] * len(fitness)) if fitness is None else np.array(fitness)
        
        ## @var rank
        # \e integer:
        # The non-domination rank of the solution (used in NSGA-II).
        self.rank = float('inf')
        
        ## @var crowding_distance
        # \e float:
        # The crowding distance of the solution (used in NSGA-II).
        self.crowding_distance = 0.0

    def dominates(self, other):
        """!
        Check if this solution dominates another solution.
        
        @param self: <em> MOParent pointer </em>
            The MOParent pointer.
        @param other: <em> MOParent object </em>
            Another solution to compare with.
            
        @return \e boolean: True if this solution dominates the other solution.
        """
        at_least_one_better = False
        for self_obj, other_obj in zip(self.fitness_values, other.fitness_values):
            if self_obj > other_obj:  # Assuming minimization
                return False
            elif self_obj < other_obj:
                at_least_one_better = True
        return at_least_one_better

    def __repr__(self):
        """!
        MOParent print function.

        @param self: <em> MOParent pointer </em>
            The MOParent pointer.
        """
        return "MOParent({}, {}, {}, {}, {}, {})".format(
            self.variables, self.fitness_values, self.rank,
            self.crowding_distance, self.changeCount, self.stallCount)

    def __str__(self):
        """!
        Human readable MOParent print function.

        @param self: <em> MOParent pointer </em>
            The MOParent pointer.
        """
        header = ["MOParent:"]
        header += ["Variables = {}".format(self.variables)]
        header += ["Fitness Values = {}".format(self.fitness_values)]
        header += ["Rank = {}".format(self.rank)]
        header += ["Crowding Distance = {}".format(self.crowding_distance)]
        header += ["Change Count = {}".format(self.changeCount)]
        header += ["Stall Count = {}".format(self.stallCount)]
        return "\n".join(header)+"\n"