import pandas as pd

criteria = pd.read_csv('data/criteria.csv')
data = pd.read_csv('data/data.csv')
decision_classes = pd.read_csv('data/decision_classes.csv')
pairwise_comparisons = pd.read_csv('data/pairwise_comp.csv')

if __name__ == '__main__':
    print(criteria)
    print(data)
    print(decision_classes)
    print(pairwise_comparisons)
