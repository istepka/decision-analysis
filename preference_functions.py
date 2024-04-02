import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class CriterionType:
    '''Enum class for criterion types.'''
    COST = 1
    GAIN = 2
    
class PreferenceFuction:
    indifference_threshold: float 
    preference_threshold: float
    criterion_type: CriterionType
    
    def __init__(self, indifference_threshold: float, preference_threshold: float, criterion_type: CriterionType):
        self.indifference_threshold = indifference_threshold
        self.preference_threshold = preference_threshold
        self.criterion_type = criterion_type
    
    def compare(self, value1: float, value2: float) -> float:
        '''Compare two values based on the preference function.'''
        if self.criterion_type == CriterionType.COST:
            return self.__compare_cost(value1, value2)
        elif self.criterion_type == CriterionType.GAIN:
            return self.__compare_gain(value1, value2)
        
    def __compare_cost(self, value1: float, value2: float) -> float:
        dist = value2 - value1
        return self._internal_compare(dist)
    
    def __compare_gain(self, value1: float, value2: float) -> float:
        dist = value1 - value2
        return self._internal_compare(dist)
    
    def _internal_compare(self, dist: float) -> float:
        raise NotImplementedError('This method should be implemented in the subclass.')
    
class VShapeWithIndifference(PreferenceFuction):
    def _internal_compare(self, dist: float) -> float:
        '''Internal method to compare two values based on the V-shape preference function.'''
        if dist > self.preference_threshold:
            return 1
        elif dist < self.indifference_threshold:
            return 0
        else:
            return (dist - self.indifference_threshold) / (self.preference_threshold - self.indifference_threshold)
        
class MarginalPreferenceMatrix:
    '''Class to store the marginal preference matrix.'''
    def __init__(self, alternatives: list[float], pfunction: PreferenceFuction, names: list[str] = None):
        self.pfunction = pfunction
        self.names = names if names is not None else [str(i) for i in range(len(alternatives))]
        self.values = alternatives
        self.matrix = np.zeros((len(self.values), len(self.values)))
        self.__compute()
        
    def __compute(self):
        '''Compute the marginal preference matrix.'''
        for i in range(len(self.values)):
            for j in range(len(self.values)):
                self.matrix[i][j] = self.pfunction.compare(self.values[i], self.values[j])
                
    def __str__(self):
        '''Nice representation of the matrix with names.'''
        df = pd.DataFrame(self.matrix, index=self.names, columns=self.names)
        return df.__str__()
    
    def __repr__(self):
        return self.__str__()
    
    def save(self, filename: str):
        '''Save the matrix to a file.'''
        df = pd.DataFrame(self.matrix, index=self.names, columns=self.names)
        df.to_csv(filename)
        

class ComprehensivePreferenceIndex:
    '''Class to compute the comprehensive preference index and positive and negative flows.'''
    
    matrix: np.ndarray
    positive_flow: np.ndarray
    negative_flow: np.ndarray     
    
    def __init__(self, mp_matrices: list[MarginalPreferenceMatrix], weights: list[float]):
        assert len(mp_matrices) == len(weights), 'The number of matrices\
            and weights should be the same, as they correspond to each other.'
        self.mp_matrices = mp_matrices
        self.weights = weights
        self.__compute()
        
    def __compute(self):
        '''Compute the comprehensive preference index.'''
        self.matrix = np.zeros((len(self.mp_matrices[0].values), len(self.mp_matrices[0].values)))
        for i, mp_matrix in enumerate(self.mp_matrices):
            self.matrix += mp_matrix.matrix * self.weights[i] / np.sum(self.weights)
            
        # Compute positive and negative flows
        self.positive_flow = np.sum(self.matrix, axis=0)
        self.negative_flow = np.sum(self.matrix, axis=1)
        
    def __str__(self):
        '''Nice representation of the matrix with names.'''
        df = pd.DataFrame(self.matrix, index=self.mp_matrices[0].names, columns=self.mp_matrices[0].names)
        return df.__str__()
    
    def __repr__(self):
        return self.__str__()
    
    def get_as_dataframe(self):
        '''Get the matrix as a pandas DataFrame.'''
        return pd.DataFrame(self.matrix, index=self.mp_matrices[0].names, columns=self.mp_matrices[0].names)
    
    def save(self, filename: str):
        '''Save the matrix to a file.'''
        df = pd.DataFrame(self.matrix, index=self.mp_matrices[0].names, columns=self.mp_matrices[0].names)
        df.to_csv(filename)
  
class PrometheeBase:
    '''Base class for the PROMETHEE methods.'''
    
    negative_ranking: pd.Series
    positive_ranking: pd.Series
    overall_ranking: pd.Series
    
    def __init__(self, cpi: ComprehensivePreferenceIndex):
        self.cpi = cpi
        self._compute()
        
    def _compute(self):
        '''Compute the PROMETHEE method.'''
        raise NotImplementedError('This method should be implemented in the subclass.')
    
    def __str__(self):
        return self.overall_ranking.__str__()
    
    def __repr__(self):
        return self.__str__()
    
    def plot_ranking(self, type: str = 'overall', savedir: str = None, show: bool = True):
        '''Plot the selected ranking (overall, positive, negative).'''
        assert type in ['overall', 'positive', 'negative'], 'The type should be either overall, positive, or negative.'
        ranking_to_use = None
        match type:
            case 'overall':
                ranking_to_use = self.overall_ranking
            case 'positive':
                ranking_to_use = self.positive_ranking
            case 'negative':
                ranking_to_use = self.negative_ranking
            case _:
                raise ValueError('This should not happen.')
        
        plt.figure(figsize=(10, 5))
        
        G = nx.DiGraph()
        # invert levels
        ranking_to_use = {name: len(ranking_to_use) - level for name, level in ranking_to_use.items()}
        for name, level in ranking_to_use.items():
            G.add_node(name, level=level)
            
        # plot
        pos = nx.multipartite_layout(G, subset_key="level", align='horizontal')
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=7, font_weight='bold')    
            
        if savedir is not None:
            plt.savefig(savedir, format='png', dpi=300)
        
        if show:
            plt.show()
            
        plt.close()
        
class PrometheeI(PrometheeBase):
    '''Class to compute the PROMETHEE I method.'''
  
    def _compute(self):
        '''Compute the PROMETHEE I method.'''
        names = self.cpi.mp_matrices[0].names
        
        # Compute the positive ranking
        positive_ranking = pd.Series(self.cpi.positive_flow, index=names).sort_index()
        
        # Compute the negative ranking
        negative_ranking = pd.Series(self.cpi.negative_flow, index=names).sort_index()

        # Build the outranking matrix
        outranking_matrix = np.zeros((len(names), len(names)))
        for i in range(len(names)):
            for j in range(len(names)):
                if j > i:
                    # Preference 
                    if positive_ranking.iloc[i] > positive_ranking.iloc[j]:
                        if negative_ranking.iloc[i] < negative_ranking.iloc[j]:
                            outranking_matrix[i][j] = 1
                        elif negative_ranking.iloc[i] == negative_ranking.iloc[j]:
                            outranking_matrix[i][j] = 1
                        else:
                            outranking_matrix[i][j] = 0
                    elif positive_ranking.iloc[i] == positive_ranking.iloc[j]:
                        if negative_ranking.iloc[i] < negative_ranking.iloc[j]:
                            outranking_matrix[i][j] = 1
                        elif negative_ranking.iloc[i] == negative_ranking.iloc[j]:
                            outranking_matrix[i][j] = 0.5
                        else:
                            outranking_matrix[i][j] = 0
                    else:
                        if negative_ranking.iloc[i] < negative_ranking.iloc[j]:
                            outranking_matrix[i][j] = 1
                        elif negative_ranking.iloc[i] == negative_ranking.iloc[j]:
                            outranking_matrix[i][j] = 0
                        else:
                            outranking_matrix[i][j] = 0
                    
                    # if outranking_matrix[i][j] == 0:
                    #     print(f'{names[i]} vs {names[j]}: {outranking_matrix[i][j]}')
                
        # Create ranking graph structure
        adjacency_list = {name: [] for name in names}
        indifferences = {name: [] for name in names}
        for i in range(len(names)):
            for j in range(len(names)):
                if outranking_matrix[i][j] == 1:
                    adjacency_list[names[i]].append(names[j])
                if outranking_matrix[i][j] == 0.5:
                    indifferences[names[i]].append(names[j])
                    
        # Add names to levels of preference relations
        # The names that are indifferent to each other are in the same level
        
        # First topology sort
        topology_sorted = []
        visited = set()
        def dfs(name):
            visited.add(name)
            for adj in adjacency_list[name]:
                if adj not in visited:
                    dfs(adj)
            topology_sorted.append(name)
            
        for name in names:
            if name not in visited:
                dfs(name)
                
        # Second topology sort to get the levels
        levels = []
        level = 0
        for name in topology_sorted:
            if name in indifferences:
                if len(indifferences[name]) > 0:
                    levels.append(level)
                else:
                    levels.append(level)
                    level += 1
            else:
                levels.append(level)
                level += 1
        
        # Create the ranking
        overall_ranking = pd.Series(index=names, data=levels).sort_values()
        
        self.overall_ranking = overall_ranking 
        self.positive_ranking = positive_ranking.astype(int) # Convert to int and naturally get the ranking
        self.negative_ranking = negative_ranking.astype(int) # Convert to int and naturally get the ranking
        
        
class PrometheeII(PrometheeBase):
    '''Class to compute the PROMETHEE II method.'''
    
    def _compute(self):
        '''Compute the PROMETHEE II method.'''
        names = self.cpi.mp_matrices[0].names
        
        # Compute the positive ranking
        positive_ranking = pd.Series(self.cpi.positive_flow, index=names).sort_index()
        
        # Compute the negative ranking
        negative_ranking = pd.Series(self.cpi.negative_flow, index=names).sort_index()
        
        # Compute the net flow
        net_flow = positive_ranking - negative_ranking
        
        # Create the ranking
        overall_ranking = net_flow.sort_values(ascending=False)
        
        self.overall_ranking = overall_ranking 
        self.positive_ranking = positive_ranking.astype(int) # Convert to int and naturally get the ranking
        self.negative_ranking = negative_ranking.astype(int) # Convert to int and naturally get the ranking
        