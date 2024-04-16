from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, LpStatus, LpMinimize
import random
import os 
import pandas as pd
import numpy as np
from collections import defaultdict
from data.parse import criteria, data, decision_classes, pairwise_comparisons_UTA


def extract_criteria_values(criteria: pd.DataFrame, data):
    criteria_values = list()
    for _, row in criteria.iterrows():
        criteria_values.append((row['Criterion'], int(row['Min Value']), int(row['Max Value']), row['Value Type']))
    return criteria_values

def get_criteria_variables(criteria_values: list[tuple]):   
    criteria_variables = defaultdict(list)
    for criterion, min_value, max_value, value_type in criteria_values:
        for i in range(min_value, max_value+1):
            criteria_variables[criterion].append(LpVariable(criterion + str(i), 0, 1))
    return criteria_variables

def add_monotonicity_constraints(problem, criteria_variables):
    # Add monotonicity constraints
    for criterion, values in criteria_variables.items():
        for i in range(len(values)-1):
            problem += values[i] <= values[i+1], "Monotonicity constraint for " + criterion + str(i)
            
def add_normalization_constraints(problem, criteria_variables):
    # Add normalization constraints
    # Lowest value for each criterion should be 0
    for criterion, values in criteria_variables.items():
        problem += values[0] == 0, "Normalization constraint for " + criterion + str(0)
    # Highest value for each criterion should together sum up to 1
    highest_values = [values[-1] for values in criteria_variables.values()]
    problem += lpSum(highest_values) == 1, "Normalization constraint for highest values"
    
def add_weight_constraints(problem, criteria_variables, max_weight=0.5, min_weight_base=0.005):
    min_weight = min_weight_base * len(criteria_variables)
    # Add weight constraints
    for criterion, values in criteria_variables.items():
        problem += values[-1] <= max_weight, "Max weight constraint for " + criterion
        problem += values[-1] >= min_weight, "Min weight constraint for " + criterion
    
    
def add_alternatives_variables(data: pd.DataFrame, criteria_variables: dict, problem: LpProblem) -> dict:
    alterntives_utilities = {}
    
    for _, row in data.iterrows():
        name = row['Movie']
        # Add variable for each alternative
        utility_var = LpVariable(name, lowBound=0)
        alterntives_utilities[name] = utility_var
        
        # Add constraint for each alternative, that its utility is the sum of the values of the criteria
        alt_vars = [criteria_variables[criterion][int(row[criterion]) - 1] for criterion in criteria_variables.keys()]
        problem += utility_var == lpSum(alt_vars), "Utility for " + name
        
    return alterntives_utilities

def add_preferential_information(problem: LpProblem, 
        pairwise_comparisons: pd.DataFrame, 
        alterntives_utilities: dict, 
        skip_vars=[],
        add_slack_vars=False
    ) -> tuple[dict, dict]:
    
    estimation_errors_vars = {}
    slack_vars = {}
    
    epsilon = 1e-4
    
    # Add over and under estimation errors for each pairwaise comparison

    for idx, (name1, name2, relation) in enumerate(pairwise_comparisons.to_numpy()):
        # print(f'Slack_{"_".join(name1.split())}_{"_".join(name2.split())}_{relation}')
        if f'Slack_{"_".join(name1.split())}_{"_".join(name2.split())}_{relation}' in skip_vars:
            print(f'Skipping {name1}-{name2}-{relation}')
            # exit()
            continue
        
        if not add_slack_vars:
            sll = LpVariable(f"UnderEstimationError_{name1}(in comp {name1}-{name2}[{idx}])", lowBound=0)
            slu = LpVariable(f"OverEstimationError_{name1}(in comp {name1}-{name2}[{idx}])", lowBound=0)
            srl = LpVariable(f"UnderEstimationError_{name2}(in comp {name1}-{name2}[{idx}])", lowBound=0)
            sru = LpVariable(f"OverEstimationError_{name2}(in comp {name1}-{name2}[{idx}])", lowBound=0)
            estimation_errors_vars[(name1, name2, idx)] = (sll, slu, srl, sru)
        else:
            slackvar = LpVariable(f"Slack_{name1}-{name2}-{relation}", lowBound=0, upBound=1, cat='Binary')
            slack_vars[(name1, name2, relation)] = slackvar
        
        if add_slack_vars:
            if relation == 'better':
                problem += alterntives_utilities[name1] >= alterntives_utilities[name2] + epsilon - slackvar, f"Preferential information for " + name1 + " and " + name2 + " " + relation
            elif relation == 'worse':
                problem += alterntives_utilities[name1] - slackvar <= alterntives_utilities[name2] - epsilon, f"Preferential information for " + name1 + " and " + name2+ " " + relation
            else:
                raise ValueError("Invalid relation in pairwise comparisons: " + relation)
        else:
            if relation == 'better':
                problem += alterntives_utilities[name1] - sll + slu >= alterntives_utilities[name2] + epsilon - srl + sru, f"Preferential information for " + name1 + " and " + name2+ " " + relation
            elif relation == 'worse':
                problem += alterntives_utilities[name1] - sll + slu <= alterntives_utilities[name2] - epsilon - srl + sru, f"Preferential information for " + name1 + " and " + name2+ " " + relation
            else:
                raise ValueError("Invalid relation in pairwise comparisons: " + relation)
        
    return estimation_errors_vars, slack_vars


def get_problem_base() -> tuple[LpProblem, dict]:
    problem = LpProblem("UTA", LpMinimize)
    
    # Extract criteria values
    criteria_values = extract_criteria_values(criteria, data)
    criteria_variables = get_criteria_variables(criteria_values)
   
    # Add constraints
    add_monotonicity_constraints(problem, criteria_variables)
    add_normalization_constraints(problem, criteria_variables)
    add_weight_constraints(problem, criteria_variables)
    
    # Add alternatives
    alt_utils_dict = add_alternatives_variables(data, criteria_variables, problem)
    
    return problem, alt_utils_dict

def UTA():
    # INCONSISTENCY ANALYSIS
    problem, alt_utils_dict = get_problem_base()
    # Add preferential information
    errors_dict, _ = add_preferential_information(problem, pairwise_comparisons_UTA, alt_utils_dict, add_slack_vars=False)
    unflatten = [e for e in errors_dict.values()]
    problem += lpSum(unflatten) 
    # Solve the problem
    problem.solve()
    
    if value(problem.objective) != 0:
        print("Inconsistency detected")
        
        
        inc_history = []
        minval = [0]
        
        solution_history = []
        
        for iteration in range(1, len(pairwise_comparisons_UTA) + 1):
            problem, alt_utils_dict = get_problem_base()
            # Add preferential information
            _, slack_vars = add_preferential_information(problem, pairwise_comparisons_UTA, alt_utils_dict, add_slack_vars=True)
            problem += lpSum(slack_vars.values())
            
            slack_vars_in_history = [var for var in slack_vars.values() if var.name in inc_history]
            problem += lpSum(slack_vars_in_history) <= max(0, minval[iteration - 1] - 1)
            
            problem.solve()
            
            print("Status:", LpStatus[problem.status])
            print("Objective value:", value(problem.objective))
            
            # Print variables
            local_solution_history = {'true': [], 'false': []}
            for var in problem.variables():
                
                if 'slack' in var.name.lower():
                    print(f"{var.name} = {value(var)}")
                    
                    if value(var) == 1:
                        inc_history.append(var.name)
                        local_solution_history['true'].append(var.name)
                    else:
                        local_solution_history['false'].append(var.name)
            
            if local_solution_history['true']:
                solution_history.append(local_solution_history)
            minval.append(len(local_solution_history['true']))
           
            if np.isclose(value(problem.objective), 0) or minval[-1] == 0:
                print("Inconsistency resolved")
                break
            
        print(f'Inconsistency history: {inc_history}')
        
        for idx, sol in enumerate(solution_history):
            print(f'Iteration {idx+1}')
            print(f'True: {sol["true"]}')
            print(f'False: {sol["false"]}')
            print()

    # Finally just do ordinal regression
    problem, alt_utils_dict = get_problem_base()
    
    # Remove inconsistent pairwise
    least_inc_to_remove = sorted(solution_history, key=lambda x: len(x['true']))[0]
    
    # Add preferential information
    errors_dict, _ = add_preferential_information(problem, pairwise_comparisons_UTA, alt_utils_dict, add_slack_vars=False, skip_vars=least_inc_to_remove['true'])
    flatten = [e for e in errors_dict.values()]
    problem += lpSum(flatten)
    
    
    
    # Solve the problem
    problem.solve()
    
    print("Status:", LpStatus[problem.status])
    print("Objective value:", value(problem.objective))
    
    # Print variables
    for var in problem.variables():
        print(f"{var.name} = {value(var)}")

    
    # Gather data for later analysis
    final_variables = {}
    for var in problem.variables():
        final_variables[var.name] = value(var)
        
    final_objective_value = value(problem.objective)
        
    return {
        'final_variables': final_variables,
        'final_objective_value': final_objective_value,
        'final_inconsistency_removed': least_inc_to_remove,
        'final_pulp_problem': problem,
        'inconsistent_solution_history': solution_history
    }


if __name__ == '__main__':
    print("UTA")
    results_dict = UTA()