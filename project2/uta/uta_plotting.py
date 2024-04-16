import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project2.uta.uta import UTA 


def plot_marginal_value_functions(utaresults: dict, criteria: pd.DataFrame, data: pd.DataFrame):
    n = len(criteria)
    variables = utaresults['final_variables']
        
    rows = 3
    cols = 3
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10), sharey=True)
    axs = axs.flatten()
    fig.suptitle('Marginal value functions')
    
    for i, criterion in enumerate(criteria['Criterion']):
        row = criteria.iloc[i]
        criterion_name = row['Criterion']
        criterion_name = '_'.join(criterion_name.split())
        low =  int(row['Min Value'])
        high = int(row['Max Value'])
        
        x = np.arange(low, high+1).astype(int)
        y = []
        
        for j in x:
            y.append(variables[criterion_name + str(j)])
            
        
        ax = axs[i]
        
        ax.plot(x, y)
        
        ax.set_title(criterion_name)
        ax.set_xlabel('Value')
        ax.set_ylabel('Marginal value')
        
    plt.tight_layout()
    plt.show()
    
def get_ranking(utaresults: dict, data: pd.DataFrame):
    variables = utaresults['final_variables']
    ranking = []
    
    for name in data['Movie']:
        name = '_'.join(name.split())
        util = variables[name]
        ranking.append((name, util))
        
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

if __name__ == '__main__':
    from data.parse import criteria, data
    results = UTA()
    
    plot_marginal_value_functions(results, criteria, data)

    ranking = get_ranking(results, data)
    print(ranking)