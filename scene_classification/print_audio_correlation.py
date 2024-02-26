import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your CSV file into a DataFrame
df = pd.read_csv('dataset.csv')
print(df)
df = df.drop(['IMAGE', 'CLASS1'], axis=1)
# Assuming 'target_variable' is the name of your target column
target_variable = 'CLASS2'

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable
df[target_variable] = label_encoder.fit_transform(df[target_variable])

# Print or inspect the DataFrame with numerical target values
print(df)

# Calculate correlation between each column and the target variable
correlations = df.corr()[target_variable]

# Print or inspect the correlations
print(correlations)

for index, row in enumerate(correlations, 1):
    print(index, row)
