import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

df = df.drop(
    columns=[
        "anaemia",
        "diabetes",
        "high_blood_pressure",
        "sex",
        "smoking",
        "time",
        "DEATH_EVENT",
    ]
)  # exclude binary features and the time feature

# print(df)
# print("Dataset shape:", df.shape)
# print(df.describe())

features = df.columns.tolist()
n_bins = 20  # ? I think it's because close to sqrt(299)...

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

for ax, feature in zip(axs.flat, features):
    ax.hist(df[feature], bins=n_bins)
    ax.set_title(feature)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()  # data before standardization

data = df.to_numpy(dtype=float)  # converting the dataframe to numpy

# print(data.shape)
# print(data[:5])

## standardization: we use z=(x-mu)/sigma for each feature/column
# data_mean = np.mean(data, axis=0)
# data_std = np.std(data, axis=0)
# data_standardized = (data - data_mean) / data_std


scaler = preprocessing.StandardScaler().fit(
    data[:150, :]
)  # fit it only computes and stores three arrays per feature

print(
    "First 150 rows means:", scaler.mean_
)  # thismean is calculated from the first 150 rows
print("First 150 rows variances:", scaler.var_)  # variance
print("First 150 rows scales:", scaler.scale_)  # sqrt(var)

data_scaled = scaler.transform(data)

print(
    "Shape after standardization", data_scaled.shape
)  # Check: Standardization changes the values not the number of observations or features

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

for i, (ax, feature) in enumerate(zip(axs.flat, features)):
    ax.hist(data_scaled[:, i], bins=n_bins)
    ax.set_title(feature)
    ax.set_xlabel("Standardized value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()  ## show standardized plt, it's the 2nd one

print("Original 299 rows means:")
print(np.mean(data, axis=0))

print("Original 299 rows standard deviations:")
print(np.std(data, axis=0, ddof=0))

print(
    "Standardized means:"
)  # sanity check for the first one: z=(x-mu)/sigma=(60.83−62.95)/12.45=−0.17 👉️ should be zero? Oh it's because it was standardized!
print(np.mean(data_scaled, axis=0))

print("Standardized standard deviations:")
print(
    np.std(data_scaled, axis=0, ddof=0)
)  # std_deviation/sigma 11.87/12.45=0.95 👉️ should be one?

scaler_all = preprocessing.StandardScaler().fit(data)
data_scaled_all = scaler_all.transform(data)

print("First 150 means:", scaler.mean_)
print("All-data means:", scaler_all.mean_)
print("First 150 variances:", scaler.var_)
print("All-data variances:", scaler_all.var_)

####----####
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

print("Minimums:", min_max_scaler.data_min_)
print("Maximums:", min_max_scaler.data_max_)

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

for i, (ax, feature) in enumerate(zip(axs.flat, features)):
    ax.hist(data_min_max_scaled[:, i], bins=n_bins)
    ax.set_title(feature)
    ax.set_xlabel("Scaled value")
    ax.set_ylabel("Frequency")

print("min of each column from fit data:", min_max_scaler.data_min_)
print("max of each column from fit data::", min_max_scaler.data_max_)
print("scale_:", min_max_scaler.scale_)
print("min_:", min_max_scaler.min_)

plt.tight_layout()
plt.show()
