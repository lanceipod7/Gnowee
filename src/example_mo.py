"""!
Example script demonstrating multi-objective optimization using MOGnowee.

This example solves a bi-objective optimization problem:
1. Minimize the spring weight (original spring objective)
2. Minimize the spring deflection

@author OpenHands AI
@date 2024
"""

import numpy as np
from MOGnowee import MOGnowee
from MOObjectiveFunction import MOObjectiveFunction
from ObjectiveFunction import ObjectiveFunction
from GnoweeUtilities import ProblemParameters

def spring_weight(u):
    """Spring weight objective function."""
    assert len(u) == 3, ('Spring design needs to specify D, W, and L and '
                         'only those 3 parameters.')
    assert u[0] != 0 and u[1] != 0 and u[2] != 0, ('Design values {} '
                                             'cannot be zero.'.format(u))
    return ((2+u[2])*u[0]**2*u[1])

def spring_deflection(u):
    """Spring deflection objective function."""
    D, W, L = u
    # Simple spring deflection formula: F*L/(G*d^4)
    F = 100  # Applied force
    G = 11.5e6  # Shear modulus
    return (F*L)/(G*D**4)

def main():
    # Create objective functions
    obj1 = ObjectiveFunction(spring_weight)
    obj2 = ObjectiveFunction(spring_deflection)
    
    # Create multi-objective function
    mo_obj = MOObjectiveFunction([obj1, obj2])
    
    # Problem parameters
    lb = [0.05, 0.25, 2.0]
    ub = [2.0, 1.3, 15.0]
    varType = ['c']*3  # All continuous variables
    
    # Create problem parameters object
    prob = ProblemParameters(
        objective=mo_obj,
        lowerBounds=lb,
        upperBounds=ub,
        varType=varType
    )
    
    # Create MOGnowee instance
    optimizer = MOGnowee(
        population=50,
        maxGens=100,
        maxFevals=5000,
        stallLimit=20,
        optConvTol=1e-6,
        pps=prob
    )
    
    # Run optimization
    timeline = optimizer.main()
    
    # Print results
    print("\nOptimization Results:")
    print("====================")
    
    # Get Pareto front solutions
    pareto_front = []
    for event in timeline:
        if isinstance(event.fitness, np.ndarray):  # Multi-objective solution
            pareto_front.append((event.design, event.fitness))
    
    print(f"\nFound {len(pareto_front)} Pareto optimal solutions:")
    for i, (design, objectives) in enumerate(pareto_front):
        print(f"\nSolution {i+1}:")
        print(f"Design variables: D={design[0]:.6f}, W={design[1]:.6f}, L={design[2]:.6f}")
        print(f"Objectives: Weight={objectives[0]:.6f}, Deflection={objectives[1]:.6f}")

if __name__ == '__main__':
    main()