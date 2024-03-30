from preference_functions import CriterionType, PreferenceFuction, VShapeWithIndifference
from parse import data, criteria, decision_classes

pfunction1 = VShapeWithIndifference(0.5, 1.0, CriterionType.GAIN)

print(pfunction1.compare(data['Music'].iloc[0], data['Music'].iloc[1]))
print(pfunction1.compare(data['Acting'].iloc[0], data['Acting'].iloc[1]))

