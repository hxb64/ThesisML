import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Chunk size for reading the dataset
SIZE = 10000
x1 = np.random.rand(SIZE)
x2 = np.random.rand(SIZE)
x3 = np.random.rand(SIZE)
x4 = np.random.rand(SIZE)
x5 = np.random.rand(SIZE)
x6 = np.random.rand(SIZE)
e = np.random.rand(SIZE)

a1 = 2
a2 = 3
a3 = 5
a4 = 7
a5 = 11
a6 = 13
y = a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6
y += e
data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6})

# Save the original data to "dataproduce.csv"
data.to_csv('/Users/Hamza/Downloads/dataproduce.csv', index=False)

trees = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
output = "Trees\tBaseLine\t0.1\t\0.01\t0.001\t0.0001\n"

results = []

for tree_count in trees:
    TRIALS = 25
    errs = [0 for i in range(5)]
    for i in range(TRIALS):
        # Shuffle records in the dataset
        data_encoded = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Splitting Dataset into train, valid, and test sets
        x = data_encoded.drop(columns=['y'], axis=1)
        y = data_encoded['y']

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
        baseline_model = lgb.LGBMRegressor(n_estimators=tree_count, verbose=-1)

        # Fitting the model without verbose parameter on the normalized training set
        baseline_model.fit(
            x_train_normalized, y_train,
        )
        errs[0] += np.sqrt(np.mean((baseline_model.predict(x_valid_normalized) - y_valid) ** 2))

        # Creating an object for the smaller GBM model and fitting it on the normalized training and validation sets
        small_model = baseline_model



        spreads = [0, 0.1, 0.01, 0.001, 0.0001]
        for j in range(1, len(spreads)):
            num_noise_vectors = 25
            for k in range(num_noise_vectors):
                # Predicting the target variable on the validation set using the smaller GBM
                y_predictions = []
                y_predictions.append(small_model.predict(x_valid_normalized))
                x_valid_noisy = x_valid_normalized + np.random.normal(0, spreads[j], x_valid_normalized.shape)
                y_predictions.append(small_model.predict(x_valid_noisy))

            # Averaging the predictions
            y_pred = np.mean(y_predictions, axis=0)

            # Evaluating the performance of the smaller GBM on the validation set
            errs[j] += np.sqrt(np.mean((y_pred - y_valid) ** 2))

    for n in range(len(errs)):
        errs[n] /= TRIALS
    print(tree_count, errs[0], errs[1], errs[2], errs[3])

    results.append([tree_count] + errs)

# Create a DataFrame from the results
columns = ['Trees', 'BaseLine', '0.1', '0.01', '0.001', '0.0001']
output_df = pd.DataFrame(results, columns=columns)

# Write the output to a CSV file
output_df.to_csv('/Users/Hamza/Downloads/datareport.csv', index=False)
