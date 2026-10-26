**Laboratory Work No. 1**
**Data Preprocessing**

**Objective:**
Familiarize yourself with data preprocessing methods from the Scikit-learn library.

**Procedure:**

**Loading Data**
1. Load the dataset from the link: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data . The data is provided as a CSV table.
2. Create a Python script. Load the dataset into a dataframe and exclude binary features and the time feature.
```python
import pandas as pd
import numpy as np
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns = 
['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
print(df) # Output the dataframe with data for the lab. There should be 299 observations and 6 features
```

3. Build histograms of the features.
```python
import matplotlib.pyplot as plt
n_bins = 20
fig, axs = plt.subplots(2,3)
axs[0, 0].hist(df['age'].values, bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(df['ejection_fraction'].values, bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(df['platelets'].values, bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(df['serum_creatinine'].values, bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(df['serum_sodium'].values, bins = n_bins)
axs[1, 2].set_title('serum_sodium')
plt.show()
```

4. Based on the histograms, determine the value ranges for each feature, as well as around which value the largest number of observations lies.
5. Since the Scikit-learn library works with NumPy arrays, convert the dataframe to a two-dimensional NumPy array, where a row corresponds to an observation and a column to a feature.
```python
data = df.to_numpy(dtype='float')
```

**Data Standardization**
1. Import the Scikit-learn module. Configure standardization based on the first 150 observations using StandardScaler.
```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(data[:150,:])
```
2. Standardize all data.
```python
data_scaled = scaler.transform(data)
```
3. Build histograms of the standardized data.
```python
fig, axs = plt.subplots(2,3)
axs[0, 0].hist(data_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
plt.show()
```
4. Compare the data before and after standardization. Describe what changed and why.
5. Calculate the mathematical expectation and standard deviation before and after standardization. Based on these values, derive for each feature the formulas by which they were standardized.
6. Compare the values from the formulas with the fields `mean_` and `var_` of the `scaler` object.
7. Perform standardization configuration on all data and compare with the results of configuration based on 150 observations.

Note: instead of the two methods `fit` and `transform`, you can use the `fit_transform` method to immediately configure parameters and transform the data.

**Scaling to a Range**
1. Scale the data to a range using MinMaxScaler.
```python
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)
```
2. Build histograms for the features and compare with the original data.
3. Using the MinMaxScaler parameters, determine the minimum and maximum value in the data for each feature.
4. Similarly, transform the data using MaxAbsScaler and RobustScaler. Build histograms. Determine to which range the data is scaled.
5. Write a function that scales all data to the range [-5, 10].

**Nonlinear Transformations**
1. Transform the data to a uniform distribution using QuantileTransformer.
```python
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, 
random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)
```
2. Build histograms and compare with the original data.
3. Determine how and what the value of the `n_quantiles` parameter affects.
4. Transform the data to a normal distribution by passing the parameter `output_distribution='normal'` to QuantileTransformer.
5. Build histograms and compare with the original data.
6. Independently transform the data to a normal distribution using PowerTransformer.

**Feature Discretization**
1. Perform discretization of the features using KBinsDiscretizer for the following number of bins:
   - age – 3
   - creatinine_phosphokinase – 4
   - ejection_fraction – 3
   - platelets – 10
   - serum_creatinine – 2
   - serum_sodium – 4
2. Build histograms. Explain the results obtained.
3. Using the `bin_edges_` parameter, output the ranges of each interval for each feature.
