import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('eda/train.csv')
unique = list(set(data['individual_id']))

label_encoder = {}
for i, value in enumerate(unique):
    label_encoder[value] = i