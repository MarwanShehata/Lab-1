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
fig.suptitle("Figure 1 Raw data distributions before standardization", fontsize=14)

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
fig.suptitle(
    "Figure 2 Standardized data with StandardScaler fit on first 150 rows; all 299 rows transformed)",
    fontsize=14,
)

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

####---- MinMaxScaler####
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

print("Minimums:", min_max_scaler.data_min_)
print("Maximums:", min_max_scaler.data_max_)

fig, axs = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Figure 3 Standardized data with MinMaxScaler",
    fontsize=14,
)
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

### MaxAbsScaler
max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)

print("Maximum absolute values:", max_abs_scaler.max_abs_)

fig, axs = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Figure 4 MaxAbsScaler transformed data", fontsize=14)

for i, (ax, feature) in enumerate(zip(axs.flat, features)):
    ax.hist(data_max_abs_scaled[:, i], bins=n_bins)
    ax.set_title(feature)
    ax.set_xlabel("Scaled value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

### RobustScaler
robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaled = robust_scaler.transform(data)

print("Medians:", robust_scaler.center_)
print("IQR scales:", robust_scaler.scale_)
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Figure 5 RobustScaler transformed data", fontsize=14)

for i, (ax, feature) in enumerate(zip(axs.flat, features)):
    ax.hist(data_robust_scaled[:, i], bins=n_bins)
    ax.set_title(feature)
    ax.set_xlabel("Robust-scaled value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()


def scale_to_range(data, new_min=-5, new_max=10):
    old_min = data.min(axis=0)
    old_max = data.max(axis=0)
    span = old_max - old_min

    # Avoid division by zero for constant features.
    safe_span = np.where(span == 0, 1, span)

    scaled = new_min + (data - old_min) / safe_span * (new_max - new_min)

    # Constant features are mapped to new_min.
    scaled[:, span == 0] = new_min

    return scaled


data_custom_scaled = scale_to_range(data)

print("Minimums:", data_custom_scaled.min(axis=0))
print("Maximums:", data_custom_scaled.max(axis=0))

### Quantile Transdormer
qt_uniform = preprocessing.QuantileTransformer(
    n_quantiles=100, output_distribution="uniform", random_state=0
).fit(data)

data_quantile_scaled = qt_uniform.transform(data)
fig.suptitle(
    "Figure 6 QuantileTransformer transformed data (uniform output)", fontsize=14
)

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

for i, (ax, feature) in enumerate(zip(axs.flat, features)):
    ax.hist(data_quantile_scaled[:, i], bins=n_bins)
    ax.set_title(feature)
    ax.set_xlabel("Quantile-transformed value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()


for n in [10, 50, 100, 299]:
    qt_uniform = preprocessing.QuantileTransformer(
        n_quantiles=n, output_distribution="uniform", random_state=0
    ).fit(data)

    transformed = qt_uniform.transform(data)

    print(
        f"Requested quantiles: {n}, " f"effective quantiles: {qt_uniform.n_quantiles_}"
    )

### QUantileTransformation Normal Distribution
qt_normal = preprocessing.QuantileTransformer(
    n_quantiles=100, output_distribution="normal", random_state=0
).fit(data)

data_quantile_normal = qt_normal.transform(data)


## PowerTransformer

pt = preprocessing.PowerTransformer(  ## based on docs we can use both  methods "yeo-johnson or box-cox" because csv data are all positive
    method="yeo-johnson", standardize=True
).fit(
    data
)

data_power_scaled = pt.transform(data)

print("Estimated lambdas:", pt.lambdas_)
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Figure 7 Power-transformed feature distributions (Yeo Johnson, standardized)",
    fontsize=14,
)
for i, (ax, feature) in enumerate(zip(axs.flat, features)):
    ax.hist(data_power_scaled[:, i], bins=n_bins)
    ax.set_title(feature)
    ax.set_xlabel("Power-transformed value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
###
## KBinsDiscretizer
