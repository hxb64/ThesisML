import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Chunk size for reading the dataset
chunk_size = 10000

# Reading the dataset in chunks
chunks = pd.read_csv('/home/hxb175/Desktop/thesis stuff/used_cars_data.csv', chunksize=chunk_size, low_memory=False)

# Initializing an empty list to store the chunks
data_chunks = []

# Iterating over the chunks and appending them to the list
for chunk in chunks:
    data_chunks.append(chunk)

# Concatenating the chunks into a single DataFrame
data = pd.concat(data_chunks, ignore_index=True)

# Shuffle records in the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Checking if 'Unnamed: 66' column exists before dropping it
if 'Unnamed: 66' in data.columns:
    data = data.drop(columns=['Unnamed: 66'], axis=1)

# Checking if 'vin' column exists before dropping it
if 'vin' in data.columns:
    data = data.drop(columns=['vin'], axis=1)

# Convert object columns to appropriate numeric types
object_cols = data.select_dtypes(include='object').columns
for col in object_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Identify categorical columns
categorical_cols = data.select_dtypes(include='object').columns

# Apply one-hot encoding to categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Splitting Dataset into train, valid, and test sets
x = data_encoded.drop(columns=['price'], axis=1)
y = data_encoded['price']
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Calculate mean and standard deviation for each feature in the training set
train_mean = x_train.mean()
train_std = x_train.std()

# Normalize the datasets using the training set mean and standard deviation
x_train_normalized = (x_train - train_mean) / train_std
x_valid_normalized = (x_valid - train_mean) / train_std
x_test_normalized = (x_test - train_mean) / train_std

# Creating an object for the baseline GBM model with early stopping
baseline_model = lgb.LGBMRegressor(n_estimators=10000, early_stopping_rounds=20)

# Fitting the model without verbose parameter on the normalized training set
baseline_model.fit(
    x_train_normalized, y_train,
    eval_set=[(x_valid_normalized, y_valid)],
    eval_metric='rmse'
)

# Get the best iteration based on validation performance
best_num_estimators = baseline_model.best_iteration_
print("Best number of estimators:", best_num_estimators)
best_iteration = baseline_model.booster_.best_iteration

# Creating an object for the smaller GBM model and fitting it on the normalized training and validation sets
############################################################int changed from 2 to 3
small_model = lgb.LGBMRegressor(n_estimators=int(best_num_estimators/1.2))

# Fitting the smaller model on the normalized data
small_model.fit(x_train_normalized, y_train, eval_set=[(x_valid_normalized, y_valid)], eval_metric='rmse')

# Predicting the target variable on the validation set using the smaller GBM
y_predictions = []
y_predictions.append(small_model.predict(x_valid_normalized))


################################################
# Adding noise to predictions and predicting on the validation set multiple times
num_noise_vectors = 20
####################
for i in range(num_noise_vectors):
    x_valid_noisy = x_valid_normalized + np.random.normal(0, 0.4, x_valid_normalized.shape)  # Adding Gaussian noise
    y_predictions.append(small_model.predict(x_valid_noisy))

# Averaging the predictions
y_pred = np.mean(y_predictions, axis=0)

# Evaluating the performance of the smaller GBM on the validation set
rmse_valid = np.sqrt(np.mean((y_pred - y_valid) ** 2))
print("Validation set RMSE (small GBM):", rmse_valid)

# Evaluating the performance of the baseline GBM on the validation set
baseline_rmse_valid = np.sqrt(np.mean((baseline_model.predict(x_valid_normalized, num_iteration=best_iteration) - y_valid) ** 2))
print("Validation set RMSE (baseline GBM):", baseline_rmse_valid)
