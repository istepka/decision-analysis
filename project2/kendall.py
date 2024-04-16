import numpy as np
import pandas as pd

def calulate_kendall_matrix(ranking: list[str], pretty_print: bool = False) -> np.ndarray | pd.DataFrame:
    # Sort by utility [1]
    sorted_ranking = sorted(ranking, key=lambda x: x[1])
    n = len(sorted_ranking)
    names, utilities = zip(*sorted_ranking)
    
    kendall_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            if utilities[i] > utilities[j]:
                kendall_matrix[i, j] = 1
            elif np.isclose(utilities[i], utilities[j]):
                kendall_matrix[i, j] = 0.5
            else:
                kendall_matrix[j, i] = 0
                
    if pretty_print:
        kendall_matrix = pd.DataFrame(kendall_matrix, index=names, columns=names)
                  
    return kendall_matrix

def calculate_kendall_distance(kendall_mat1, kendall_mat2):
    assert kendall_mat1.shape == kendall_mat2.shape, "Matrices must have the same shape"
    return 0.5 * np.sum(np.abs(kendall_mat1 - kendall_mat2))

def compute_kendalls_tau(ranking1: list[str], ranking2: list[str]) -> float:
    kendall_mat1 = calulate_kendall_matrix(ranking1)
    kendall_mat2 = calulate_kendall_matrix(ranking2)
    
    dist = calculate_kendall_distance(kendall_mat1, kendall_mat2)
    tau = 1 - 4 * (dist / (len(ranking1) * (len(ranking1) - 1)))
    return tau
    
            