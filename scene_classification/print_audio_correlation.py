import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_correlation_values_sorted():
    df = pd.read_csv('dataset.csv')
    df = df.drop(['IMAGE', 'CLASS1'], axis=1)
    # Assuming 'target_variable' is the name of your target column
    target_variable = 'CLASS2'

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the target variable
    df[target_variable] = label_encoder.fit_transform(df[target_variable])

    # Calculate correlation between each column and the target variable
    correlations = df.corr()[target_variable]

    correltation_indexes = [[i, val] for i, val in enumerate(correlations, 1)]
    correltation_indexes.sort(key=lambda x: x[1], reverse=True)
    correltation_indexes = correltation_indexes[1:]
    return correltation_indexes

if __name__ == '__main__':
    correltation_indexes = get_correlation_values_sorted()
    print(correltation_indexes)
    for i in range(10):
        print(correltation_indexes[i])
    # for index, row in enumerate(correlations, 1):
    #     print(index, row)
    
