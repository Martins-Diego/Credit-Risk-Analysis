# Credit Risk Analysis - EDA


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
df = pd.read_csv("C:/Users/Diego/Documents/Data_Analytics/Python/Credit_Risk_Management/credit_risk_dataset.csv")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_age</th>
      <th>person_income</th>
      <th>person_home_ownership</th>
      <th>person_emp_length</th>
      <th>loan_intent</th>
      <th>loan_grade</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_default_on_file</th>
      <th>cb_person_cred_hist_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>59000</td>
      <td>RENT</td>
      <td>123.0</td>
      <td>PERSONAL</td>
      <td>D</td>
      <td>35000</td>
      <td>16.02</td>
      <td>1</td>
      <td>0.59</td>
      <td>Y</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>9600</td>
      <td>OWN</td>
      <td>5.0</td>
      <td>EDUCATION</td>
      <td>B</td>
      <td>1000</td>
      <td>11.14</td>
      <td>0</td>
      <td>0.10</td>
      <td>N</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>9600</td>
      <td>MORTGAGE</td>
      <td>1.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>5500</td>
      <td>12.87</td>
      <td>1</td>
      <td>0.57</td>
      <td>N</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>65500</td>
      <td>RENT</td>
      <td>4.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>35000</td>
      <td>15.23</td>
      <td>1</td>
      <td>0.53</td>
      <td>N</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>54400</td>
      <td>RENT</td>
      <td>8.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>35000</td>
      <td>14.27</td>
      <td>1</td>
      <td>0.55</td>
      <td>Y</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32576</th>
      <td>57</td>
      <td>53000</td>
      <td>MORTGAGE</td>
      <td>1.0</td>
      <td>PERSONAL</td>
      <td>C</td>
      <td>5800</td>
      <td>13.16</td>
      <td>0</td>
      <td>0.11</td>
      <td>N</td>
      <td>30</td>
    </tr>
    <tr>
      <th>32577</th>
      <td>54</td>
      <td>120000</td>
      <td>MORTGAGE</td>
      <td>4.0</td>
      <td>PERSONAL</td>
      <td>A</td>
      <td>17625</td>
      <td>7.49</td>
      <td>0</td>
      <td>0.15</td>
      <td>N</td>
      <td>19</td>
    </tr>
    <tr>
      <th>32578</th>
      <td>65</td>
      <td>76000</td>
      <td>RENT</td>
      <td>3.0</td>
      <td>HOMEIMPROVEMENT</td>
      <td>B</td>
      <td>35000</td>
      <td>10.99</td>
      <td>1</td>
      <td>0.46</td>
      <td>N</td>
      <td>28</td>
    </tr>
    <tr>
      <th>32579</th>
      <td>56</td>
      <td>150000</td>
      <td>MORTGAGE</td>
      <td>5.0</td>
      <td>PERSONAL</td>
      <td>B</td>
      <td>15000</td>
      <td>11.48</td>
      <td>0</td>
      <td>0.10</td>
      <td>N</td>
      <td>26</td>
    </tr>
    <tr>
      <th>32580</th>
      <td>66</td>
      <td>42000</td>
      <td>RENT</td>
      <td>2.0</td>
      <td>MEDICAL</td>
      <td>B</td>
      <td>6475</td>
      <td>9.99</td>
      <td>0</td>
      <td>0.15</td>
      <td>N</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
<p>32581 rows × 12 columns</p>
</div>




```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>person_age</th>
      <td>32581.0</td>
      <td>27.734600</td>
      <td>6.348078</td>
      <td>20.00</td>
      <td>23.00</td>
      <td>26.00</td>
      <td>30.00</td>
      <td>144.00</td>
    </tr>
    <tr>
      <th>person_income</th>
      <td>32581.0</td>
      <td>66074.848470</td>
      <td>61983.119168</td>
      <td>4000.00</td>
      <td>38500.00</td>
      <td>55000.00</td>
      <td>79200.00</td>
      <td>6000000.00</td>
    </tr>
    <tr>
      <th>person_emp_length</th>
      <td>31686.0</td>
      <td>4.789686</td>
      <td>4.142630</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>7.00</td>
      <td>123.00</td>
    </tr>
    <tr>
      <th>loan_amnt</th>
      <td>32581.0</td>
      <td>9589.371106</td>
      <td>6322.086646</td>
      <td>500.00</td>
      <td>5000.00</td>
      <td>8000.00</td>
      <td>12200.00</td>
      <td>35000.00</td>
    </tr>
    <tr>
      <th>loan_int_rate</th>
      <td>29465.0</td>
      <td>11.011695</td>
      <td>3.240459</td>
      <td>5.42</td>
      <td>7.90</td>
      <td>10.99</td>
      <td>13.47</td>
      <td>23.22</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>32581.0</td>
      <td>0.218164</td>
      <td>0.413006</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>loan_percent_income</th>
      <td>32581.0</td>
      <td>0.170203</td>
      <td>0.106782</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.15</td>
      <td>0.23</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>cb_person_cred_hist_length</th>
      <td>32581.0</td>
      <td>5.804211</td>
      <td>4.055001</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>8.00</td>
      <td>30.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Output the duplicates
df[df.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_age</th>
      <th>person_income</th>
      <th>person_home_ownership</th>
      <th>person_emp_length</th>
      <th>loan_intent</th>
      <th>loan_grade</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_default_on_file</th>
      <th>cb_person_cred_hist_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15975</th>
      <td>23</td>
      <td>42000</td>
      <td>RENT</td>
      <td>5.0</td>
      <td>VENTURE</td>
      <td>B</td>
      <td>6000</td>
      <td>9.99</td>
      <td>0</td>
      <td>0.14</td>
      <td>N</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15989</th>
      <td>23</td>
      <td>90000</td>
      <td>MORTGAGE</td>
      <td>7.0</td>
      <td>EDUCATION</td>
      <td>B</td>
      <td>8000</td>
      <td>10.36</td>
      <td>0</td>
      <td>0.09</td>
      <td>N</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15995</th>
      <td>24</td>
      <td>48000</td>
      <td>MORTGAGE</td>
      <td>4.0</td>
      <td>MEDICAL</td>
      <td>A</td>
      <td>4000</td>
      <td>5.42</td>
      <td>0</td>
      <td>0.08</td>
      <td>N</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16025</th>
      <td>24</td>
      <td>10000</td>
      <td>RENT</td>
      <td>8.0</td>
      <td>PERSONAL</td>
      <td>A</td>
      <td>3000</td>
      <td>7.90</td>
      <td>1</td>
      <td>0.30</td>
      <td>N</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16028</th>
      <td>23</td>
      <td>100000</td>
      <td>MORTGAGE</td>
      <td>7.0</td>
      <td>EDUCATION</td>
      <td>A</td>
      <td>15000</td>
      <td>7.88</td>
      <td>0</td>
      <td>0.15</td>
      <td>N</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32010</th>
      <td>42</td>
      <td>39996</td>
      <td>MORTGAGE</td>
      <td>2.0</td>
      <td>HOMEIMPROVEMENT</td>
      <td>A</td>
      <td>2500</td>
      <td>5.42</td>
      <td>0</td>
      <td>0.06</td>
      <td>N</td>
      <td>12</td>
    </tr>
    <tr>
      <th>32047</th>
      <td>36</td>
      <td>250000</td>
      <td>RENT</td>
      <td>2.0</td>
      <td>DEBTCONSOLIDATION</td>
      <td>A</td>
      <td>20000</td>
      <td>7.88</td>
      <td>0</td>
      <td>0.08</td>
      <td>N</td>
      <td>17</td>
    </tr>
    <tr>
      <th>32172</th>
      <td>49</td>
      <td>120000</td>
      <td>MORTGAGE</td>
      <td>12.0</td>
      <td>MEDICAL</td>
      <td>B</td>
      <td>12000</td>
      <td>10.99</td>
      <td>0</td>
      <td>0.10</td>
      <td>N</td>
      <td>12</td>
    </tr>
    <tr>
      <th>32259</th>
      <td>39</td>
      <td>40000</td>
      <td>OWN</td>
      <td>4.0</td>
      <td>VENTURE</td>
      <td>B</td>
      <td>1000</td>
      <td>10.37</td>
      <td>0</td>
      <td>0.03</td>
      <td>N</td>
      <td>16</td>
    </tr>
    <tr>
      <th>32279</th>
      <td>43</td>
      <td>11340</td>
      <td>RENT</td>
      <td>4.0</td>
      <td>EDUCATION</td>
      <td>C</td>
      <td>1950</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.17</td>
      <td>N</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>165 rows × 12 columns</p>
</div>




```python
# Remove duplicates (boolean indexing)
df = df[~df.duplicated()]
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_age</th>
      <th>person_income</th>
      <th>person_home_ownership</th>
      <th>person_emp_length</th>
      <th>loan_intent</th>
      <th>loan_grade</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_default_on_file</th>
      <th>cb_person_cred_hist_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>59000</td>
      <td>RENT</td>
      <td>123.0</td>
      <td>PERSONAL</td>
      <td>D</td>
      <td>35000</td>
      <td>16.02</td>
      <td>1</td>
      <td>0.59</td>
      <td>Y</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>9600</td>
      <td>OWN</td>
      <td>5.0</td>
      <td>EDUCATION</td>
      <td>B</td>
      <td>1000</td>
      <td>11.14</td>
      <td>0</td>
      <td>0.10</td>
      <td>N</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>9600</td>
      <td>MORTGAGE</td>
      <td>1.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>5500</td>
      <td>12.87</td>
      <td>1</td>
      <td>0.57</td>
      <td>N</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>65500</td>
      <td>RENT</td>
      <td>4.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>35000</td>
      <td>15.23</td>
      <td>1</td>
      <td>0.53</td>
      <td>N</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>54400</td>
      <td>RENT</td>
      <td>8.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>35000</td>
      <td>14.27</td>
      <td>1</td>
      <td>0.55</td>
      <td>Y</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32576</th>
      <td>57</td>
      <td>53000</td>
      <td>MORTGAGE</td>
      <td>1.0</td>
      <td>PERSONAL</td>
      <td>C</td>
      <td>5800</td>
      <td>13.16</td>
      <td>0</td>
      <td>0.11</td>
      <td>N</td>
      <td>30</td>
    </tr>
    <tr>
      <th>32577</th>
      <td>54</td>
      <td>120000</td>
      <td>MORTGAGE</td>
      <td>4.0</td>
      <td>PERSONAL</td>
      <td>A</td>
      <td>17625</td>
      <td>7.49</td>
      <td>0</td>
      <td>0.15</td>
      <td>N</td>
      <td>19</td>
    </tr>
    <tr>
      <th>32578</th>
      <td>65</td>
      <td>76000</td>
      <td>RENT</td>
      <td>3.0</td>
      <td>HOMEIMPROVEMENT</td>
      <td>B</td>
      <td>35000</td>
      <td>10.99</td>
      <td>1</td>
      <td>0.46</td>
      <td>N</td>
      <td>28</td>
    </tr>
    <tr>
      <th>32579</th>
      <td>56</td>
      <td>150000</td>
      <td>MORTGAGE</td>
      <td>5.0</td>
      <td>PERSONAL</td>
      <td>B</td>
      <td>15000</td>
      <td>11.48</td>
      <td>0</td>
      <td>0.10</td>
      <td>N</td>
      <td>26</td>
    </tr>
    <tr>
      <th>32580</th>
      <td>66</td>
      <td>42000</td>
      <td>RENT</td>
      <td>2.0</td>
      <td>MEDICAL</td>
      <td>B</td>
      <td>6475</td>
      <td>9.99</td>
      <td>0</td>
      <td>0.15</td>
      <td>N</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
<p>32416 rows × 12 columns</p>
</div>




```python
# Check for outliers using Interquartile Range Method (IQR) 
def find_outliers(df, col):
    # Calculate Q1, Q3, and IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    # Determine the lower and upper bounds for outliers
    return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
```


```python
numerical_columns = ["person_age","person_income","person_emp_length","cb_person_cred_hist_length"]
for col in numerical_columns:
    print(find_outliers(df, col).shape)
```

    (1491, 12)
    (1478, 12)
    (852, 12)
    (1139, 12)
    


```python
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_range = Q1 - 1.5*IQR
    upper_range = Q3 + 1.5*IQR
    df[col] = np.where(df[col]>upper_range, upper_range, df[col])
    df[col] = np.where(df[col]<lower_range, lower_range, df[col])
    return df
```


```python
for col in numerical_columns:
    remove_outliers(df, col)
```


```python
for col in numerical_columns:
    remove_outliers(df_copy, col)
```


```python
# Drop null values 
df = df.dropna()
```


```python
# Loan status distribution
plt.figure(figsize=(6, 5))
ax = sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Frequency')

# Add Bar counts
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

plt.show()
```


    
![png](output_14_0.png)
    



```python
# Person Age vs Person Income
plt.figure(figsize=(8, 6))
plt.hexbin(df['person_age'], df['person_income'], gridsize=20, cmap='viridis')
plt.title('Person Age vs. Person Income')
plt.xlabel('Person Age')
plt.ylabel('Person Income')
cb = plt.colorbar()
cb.set_label('Data Density')
plt.show()
```


    
![png](output_15_0.png)
    



```python
# Loan amount by loan intention
plt.figure(figsize=(12, 6))
sns.boxplot(x='loan_intent', y='loan_amnt', data=df)
plt.title('Loan Amount Distribution by Loan Purpose')
plt.xlabel('Loan Purpose')
plt.ylabel('Loan Amount')
plt.show()
```


    
![png](output_16_0.png)
    



```python
# Loan Status vs Age
plt.figure(figsize=[20,10])
sns.countplot(x = 'person_age', hue= 'loan_status', data=df);
```


    
![png](output_17_0.png)
    



```python
# Average loan_int_rate by loan_grade
plt.figure(figsize=(12, 6))
sns.barplot(x='loan_grade', y='loan_int_rate', data=df, order=sorted(df['loan_grade'].unique()), ci=None)
plt.title('Average Loan Interest Rate by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Average Loan Interest Rate')
plt.show()
```


    
![png](output_18_0.png)
    



```python
# Create a correlation matrix
correlations = df[["person_age", "person_income", "person_emp_length", "cb_person_cred_hist_length"]].corr()
# Create a heatmap to visualize the correlation matrix
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(correlations, annot=True, cmap="coolwarm", square=True)
# Display the correlation matrix with rounded values
correlations.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>cb_person_cred_hist_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>person_age</th>
      <td>1.00</td>
      <td>0.13</td>
      <td>0.16</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>person_income</th>
      <td>0.13</td>
      <td>1.00</td>
      <td>0.21</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>person_emp_length</th>
      <td>0.16</td>
      <td>0.21</td>
      <td>1.00</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>cb_person_cred_hist_length</th>
      <td>0.87</td>
      <td>0.10</td>
      <td>0.13</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_19_1.png)
    



```python
# Loan Defaulters Pie Chart
loan_defaulters = df[df.loan_status == 1].loan_status.count()
non_defaulters = df[df.loan_status == 0].loan_status.count()
values = [loan_defaulters, non_defaulters]
colors = ['r', 'b']
explode = [0, 0.1]
labels = ['Loan Defaulters', 'Non-Defaulters']
plt.pie(values, colors=colors, labels=labels, explode=explode)
plt.title('Representation of Loan Defaulters and Non-Defaulters')
plt.show()
```


    
![png](output_20_0.png)
    



```python

```
