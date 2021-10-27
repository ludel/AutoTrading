import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../data/cac_merge.csv')

# g = data.groupby('Day').count() != 34
# g.loc[g['Open']].index.to_list()
exclude = [342, 376, 606, 1534, 1756, 1757, 1758, 1759, 1760, 1859, 2048, 2386, 2387, 2389, 2829, 2830, 2831, 3836,
           4119, 4120, 4204]
data = data.loc[~data['Day'].isin(exclude)]

data['Day'] = data.groupby('Date').ngroup()

data = data.sort_values(['Day', 'Code'])
data = data.set_index('Day')

data = data.loc[328:]

data['Code_number'] = LabelEncoder().fit_transform(data['Code'])
data = data.drop('Date', axis=1)
data.to_csv('../data/cac_merge_clean.csv')
