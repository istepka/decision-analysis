import numpy as np 

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
        