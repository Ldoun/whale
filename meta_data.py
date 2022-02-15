import pandas as pd

data = pd.read_csv('train.csv')
unique = list(set(data['individual_id']))

label_encoder = {}
for i, value in enumerate(unique):
    label_encoder[value] = i