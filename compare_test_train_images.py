import pandas as pd
import numpy as np

train_data = pd.read_csv('')
test_data = pd.read_csv('')

bs = 100
logit_scale = 14.2857

train_vector_values = []
test_vector_values = []
train_whale_id = []

for _, train_row,test_row in zip(train_data.iterrows(), test_data.iterrows()):
    train_array = np.load(train_row['npy'])
    test_array = np.load(test_row['npy'])
    train_vector_values.append(train_array)
    test_vector_values.append(test_array)
    
all_train_vector = np.stack(train_vector_values)
all_test_vector = np.stack(test_vector_values)

cosine_similarity = logit_scale * all_test_vector @ all_train_vector

prediction = [train_whale_id[index] for index in np.argmax(cosine_similarity,axis=0)]
