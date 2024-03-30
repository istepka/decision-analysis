import pandas as pd

criteria = pd.read_csv('criteria.csv')
data = pd.read_csv('data.csv')
decision_classes = pd.read_csv('decision_classes.csv')

if __name__ == '__main__':
    print(criteria)
    print(data)
    print(decision_classes)
