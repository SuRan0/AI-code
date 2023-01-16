import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend import assoc

# Load the dataset
df = pd.read_csv('dataset.csv')

# Perform Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generate the association ruls
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Print the result
print(frequent_itemsets)
print(rules)