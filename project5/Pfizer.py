from pulp import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

### distance matrix
D = [
    [16.160, 24.080, 24.320, 21.120],
    [19.000, 26.470, 27.240, 17.330],
    [25.290, 32.490, 33.420, 12.250],
    [0.000, 7.930, 8.310, 36.120],
    [3.070, 6.440, 7.560, 37.360],
    [1.220, 7.510, 8.190, 36.290],
    [2.800, 10.310, 10.950, 33.500],
    [2.870, 5.070, 5.670, 38.800],
    [3.800, 8.010, 7.410, 38.160],
    [12.350, 4.520, 4.350, 48.270],
    [11.110, 3.480, 2.970, 47.140],
    [21.990, 22.020, 24.070, 39.860],
    [8.820, 3.300, 5.360, 43.310],
    [7.930, 0.000, 2.070, 43.750],
    [9.340, 2.250, 1.110, 45.430],
    [8.310, 2.070, 0.000, 44.430],
    [7.310, 2.440, 1.110, 43.430],
    [7.550, 0.750, 1.530, 43.520],
    [11.130, 18.410, 19.260, 25.400],
    [17.490, 23.440, 24.760, 23.210],
    [11.030, 18.930, 19.280, 25.430],
    [36.120, 43.750, 44.430, 0.000]
]

### labor intensity
P = [0.1609, 0.1164, 0.1026, 0.1516, 0.0939, 0.1320, 0.0687, 0.0930, 0.2116, 0.2529, 0.0868, 0.0828, 0.0975, 0.8177,
     0.4115, 0.3795, 0.0710, 0.0427, 0.1043, 0.0997, 0.1698, 0.2531]

### current assignment
A = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]

### current locations of representatives
L = [4, 14, 16, 22]


### calculations
def solve_for_epsilon(epsilon_f2):
    no_representatives = len(L)
    no_regions = len(D)
    pfizer_lp_problem = LpProblem("PfizerTurkey", LpMinimize)

    # Create binary variables
    binary_vars = [[LpVariable(f'x_{repr}_{reg}', cat="Binary") for reg in range(
        no_regions)] for repr in range(no_representatives)]
    
    # No region should be assigned to more than one representative
    for region in range(no_regions):
        pfizer_lp_problem += lpSum([vars[region] for vars in binary_vars]) == 1
        
    # Intensity should have sum [0.9, 1.1]
    for vars in binary_vars:
        total_intensity = lpSum(
            [var * intensity for var, intensity in zip(vars, P)])
        pfizer_lp_problem += total_intensity >= 0.9
        pfizer_lp_problem += total_intensity <= 1.1

    # F1 definition - minimize the distance
    f1_objective = 0
    for i, vars in enumerate(binary_vars):
        f1_objective += lpSum(var * distance for var,
                               distance in zip(vars, np.array(D).T[i]))
        
    # Set the objective
    pfizer_lp_problem.setObjective(f1_objective)

    # F2 definition - minimize the labor change
    f2_objective = 0
    assignment_array = np.array(A).T
    for i, vars in enumerate(binary_vars):
        new_idx = np.where(assignment_array[i] == 0)[0]
        f2_objective += lpSum([vars[j] * P[j] for j in new_idx])

    pfizer_lp_problem += f2_objective <= epsilon_f2

    # Optimize
    pfizer_lp_problem.solve(PULP_CBC_CMD(msg=0))

    # If optimal, return the solution, otherwise return -1
    if LpStatus[pfizer_lp_problem.status] == "Optimal":
        solution = np.array([[var.value() for var in row] for row in binary_vars])
        return pfizer_lp_problem.objective.value(), f2_objective.value(), solution
    else:
        return -1, epsilon_f2, None
    
    

if __name__ == '__main__':
    labor_change_upper_bound = sum(P)

    print(f"Current F1 value: {(np.array(A) * np.array(D)).sum()}")
    print(f"F1 lower bound: {np.array(D).min(axis=1).sum()}")
    print(f"Labor change upper bound: {labor_change_upper_bound}")

    results = [solve_for_epsilon(i) for i in tqdm(np.linspace(0.1, labor_change_upper_bound, 200))]
    solutions_objective_values = [s for _, _, s in results]
    solutions_coordinates_map = np.array([(x, y) for x, y, _ in results]) # round maybe?

    map_point_to_solution = {tuple(solutions_coordinates_map[i]): solutions_objective_values[i]
                         for i in range(len(solutions_objective_values))}

    solutions_coordinates_map = solutions_coordinates_map[solutions_coordinates_map[:, 0] >= 0].T

    pareto_frontier = sorted({tuple(point) for point in solutions_coordinates_map.T if not any(
        (solutions_coordinates_map[0] <= point[0]) & (solutions_coordinates_map[1] < point[1]))})

    print("Pareto:", len(pareto_frontier))
    for pareto_point in pareto_frontier:
        print(f"\n\nSolution for F1={pareto_point[0]}, F2={pareto_point[1]}")
        print(map_point_to_solution[pareto_point])

    x, y = zip(*pareto_frontier)

    utopian_point = (min(x), min(y))
    nadir_point = (max(x), max(y))
    max_f1 = max(x)
    max_f2 = max(y)
    min_f1 = min(x)
    min_f2 = min(y)

    plt.scatter(x, y, label="Pareto Points")
    plt.scatter(*utopian_point, color='red', label='Utopian Point')
    plt.scatter(*nadir_point, color='green', label='Nadir Point')
    plt.axhline(max_f2, color='gray', linestyle='--', label='_Max F2 Boundary')
    plt.axvline(max_f1, color='gray', linestyle='--', label='_Max F1 Boundary')

    plt.axhline(min_f2, color='gray', linestyle='--', label='_Min F2 Boundary')
    plt.axvline(min_f1, color='gray', linestyle='--', label='_Min F1 Boundary')

    plt.plot([min_f1, max_f1], [min_f2, min_f2], 'k--', lw=1)
    plt.plot([min_f1, min_f1], [min_f2, max_f2], 'k--', lw=1)
    
    plt.annotate('Utopian', utopian_point, textcoords="offset points", xytext=(0,10), ha='right', color='red')
    plt.annotate('Nadir', nadir_point, textcoords="offset points", xytext=(0,10), ha='left', color='green')
    plt.annotate(f'Max F1: {max_f1}', (max_f1, max_f2), textcoords="offset points", xytext=(-40,-10), ha='center', color='gray')
    plt.annotate(f'Max F2: {max_f2}', (max_f1, max_f2), textcoords="offset points", xytext=(-40,-20), ha='center', color='gray')

    plt.xlabel('F1 Value')
    plt.ylabel('F2 Value')
    plt.savefig('pfizer_turkey_frontier.pdf', bbox_inches='tight')