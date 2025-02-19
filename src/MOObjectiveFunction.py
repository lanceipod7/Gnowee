"""!
@file src/MOObjectiveFunction.py
@package Gnowee

@defgroup MOObjectiveFunction MOObjectiveFunction

@brief Extension of ObjectiveFunction class to support multiple objectives.

This class extends the ObjectiveFunction class to handle multiple objective
functions for multi-objective optimization.

@author OpenHands AI

@date 2024

@copyright GNU GPLv3.0+
"""

from ObjectiveFunction import ObjectiveFunction
import numpy as np

class MOObjectiveFunction:
    """!
    @ingroup MOObjectiveFunction
    Class for handling multiple objective functions.
    """

    def __init__(self, objectives=None):
        """!
        Constructor to build the MOObjectiveFunction class.

        @param self: <em> MOObjectiveFunction pointer </em>
            The MOObjectiveFunction pointer.
        @param objectives: <em> list of ObjectiveFunction objects </em>
            List of objective functions to be optimized.
        """
        ## @var objectives
        # <em> list of ObjectiveFunction objects: </em>
        # The list of objective functions to be optimized.
        self.objectives = objectives if objectives is not None else []

    def add_objective(self, objective):
        """!
        Add an objective function to the list.
        
        @param self: <em> MOObjectiveFunction pointer </em>
            The MOObjectiveFunction pointer.
        @param objective: <em> ObjectiveFunction object </em>
            The objective function to add.
        """
        if isinstance(objective, ObjectiveFunction):
            self.objectives.append(objective)
        else:
            raise TypeError("Objective must be an ObjectiveFunction instance")

    def evaluate(self, u):
        """!
        Evaluate all objective functions for given design variables.
        
        @param self: <em> MOObjectiveFunction pointer </em>
            The MOObjectiveFunction pointer.
        @param u: \e array
            The design parameters to be evaluated.
            
        @return \e array: Array of fitness values for each objective.
        """
        return np.array([obj.func(u) for obj in self.objectives])

    def __repr__(self):
        """!
        MOObjectiveFunction print function.

        @param self: <em> MOObjectiveFunction pointer </em>
            The MOObjectiveFunction pointer.
        """
        return "MOObjectiveFunction({})".format(self.objectives)

    def __str__(self):
        """!
        Human readable MOObjectiveFunction print function.

        @param self: <em> MOObjectiveFunction pointer </em>
            The MOObjectiveFunction pointer.
        """
        header = ["MOObjectiveFunction:"]
        for i, obj in enumerate(self.objectives):
            header += ["Objective {}: {}".format(i+1, obj)]
        return "\n".join(header)+"\n"