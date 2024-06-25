```python
import pandas as pd, numpy as np
```


```python

import os
os.chdir('C:\\Users\\Home\\Downloads')
```


```python
df_ipl = pd.read_csv("IPL_ball_by_ball_updated till 2024.csv",low_memory=False)
salary = pd.read_excel("IPL SALARIES 2024.xlsx")
```


```python
df_ipl.columns
```




    Index(['Match id', 'Date', 'Season', 'Batting team', 'Bowling team',
           'Innings No', 'Ball No', 'Bowler', 'Striker', 'Non Striker',
           'runs_scored', 'extras', 'type of extras', 'score', 'score/wicket',
           'wicket_confirmation', 'wicket_type', 'fielders_involved',
           'Player Out'],
          dtype='object')




```python
grouped_data = df_ipl.groupby(['Season', 'Innings No', 'Striker','Bowler']).agg({'runs_scored': sum, 'wicket_confirmation':sum}).reset_index()
```


```python

grouped_data
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
      <th>Season</th>
      <th>Innings No</th>
      <th>Striker</th>
      <th>Bowler</th>
      <th>runs_scored</th>
      <th>wicket_confirmation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007/08</td>
      <td>1</td>
      <td>A Chopra</td>
      <td>DP Vijaykumar</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007/08</td>
      <td>1</td>
      <td>A Chopra</td>
      <td>DW Steyn</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007/08</td>
      <td>1</td>
      <td>A Chopra</td>
      <td>GD McGrath</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007/08</td>
      <td>1</td>
      <td>A Chopra</td>
      <td>PJ Sangwan</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007/08</td>
      <td>1</td>
      <td>A Chopra</td>
      <td>RP Singh</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>48781</th>
      <td>2024</td>
      <td>2</td>
      <td>YBK Jaiswal</td>
      <td>RJW Topley</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48782</th>
      <td>2024</td>
      <td>2</td>
      <td>YBK Jaiswal</td>
      <td>SM Curran</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48783</th>
      <td>2024</td>
      <td>2</td>
      <td>YBK Jaiswal</td>
      <td>Tilak Varma</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48784</th>
      <td>2024</td>
      <td>2</td>
      <td>YBK Jaiswal</td>
      <td>VG Arora</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48785</th>
      <td>2024</td>
      <td>2</td>
      <td>YBK Jaiswal</td>
      <td>Yash Thakur</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>48786 rows × 6 columns</p>
</div>




```python
total_runs_each_year = grouped_data.groupby(['Season', 'Striker'])['runs_scored'].sum().reset_index()
total_wicket_each_year = grouped_data.groupby(['Season', 'Bowler'])['wicket_confirmation'].sum().reset_index()
```


```python

total_runs_each_year
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
      <th>Season</th>
      <th>Striker</th>
      <th>runs_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007/08</td>
      <td>A Chopra</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007/08</td>
      <td>A Kumble</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007/08</td>
      <td>A Mishra</td>
      <td>37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007/08</td>
      <td>A Mukund</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007/08</td>
      <td>A Nehra</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2593</th>
      <td>2024</td>
      <td>Vijaykumar Vyshak</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2594</th>
      <td>2024</td>
      <td>WG Jacks</td>
      <td>176</td>
    </tr>
    <tr>
      <th>2595</th>
      <td>2024</td>
      <td>WP Saha</td>
      <td>135</td>
    </tr>
    <tr>
      <th>2596</th>
      <td>2024</td>
      <td>Washington Sundar</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2597</th>
      <td>2024</td>
      <td>YBK Jaiswal</td>
      <td>249</td>
    </tr>
  </tbody>
</table>
<p>2598 rows × 3 columns</p>
</div>




```python
#pip install python-Levenshtein
```


```python
pip install python-Levenshtein
```

    Requirement already satisfied: python-Levenshtein in c:\anaconda\lib\site-packages (0.25.1)
    Requirement already satisfied: Levenshtein==0.25.1 in c:\anaconda\lib\site-packages (from python-Levenshtein) (0.25.1)
    Requirement already satisfied: rapidfuzz<4.0.0,>=3.8.0 in c:\anaconda\lib\site-packages (from Levenshtein==0.25.1->python-Levenshtein) (3.9.3)
    Note: you may need to restart the kernel to use updated packages.
    


```python
from fuzzywuzzy import process

# Convert to DataFrame
df_salary = salary.copy()
df_runs = total_runs_each_year.copy()

# Function to match names
def match_names(name, names_list):
    match, score = process.extractOne(name, names_list)
    return match if score >= 80 else None  # Use a threshold score of 80

# Create a new column in df_salary with matched names from df_runs
df_salary['Matched_Player'] = df_salary['Player'].apply(lambda x: match_names(x, df_runs['Striker'].tolist()))

# Merge the DataFrames on the matched names
df_merged = pd.merge(df_salary, df_runs, left_on='Matched_Player', right_on='Striker')
```


```python
df_original = df_merged.copy()
```


```python
#susbsets data for last three years
df_merged = df_merged.loc[df_merged['Season'].isin(['2021', '2022', '2023'])]
```


```python
df_merged.Season.unique()
```




    array(['2023', '2022', '2021'], dtype=object)




```python
df_merged.Season.unique()

```




    array(['2023', '2022', '2021'], dtype=object)




```python

df_merged.head()
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
      <th>Player</th>
      <th>Salary</th>
      <th>Rs</th>
      <th>international</th>
      <th>iconic</th>
      <th>Matched_Player</th>
      <th>Season</th>
      <th>Striker</th>
      <th>runs_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abhishek Porel</td>
      <td>20 lakh</td>
      <td>20</td>
      <td>0</td>
      <td>NaN</td>
      <td>Abishek Porel</td>
      <td>2023</td>
      <td>Abishek Porel</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anrich Nortje</td>
      <td>6.5 crore</td>
      <td>650</td>
      <td>1</td>
      <td>NaN</td>
      <td>A Nortje</td>
      <td>2022</td>
      <td>A Nortje</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Anrich Nortje</td>
      <td>6.5 crore</td>
      <td>650</td>
      <td>1</td>
      <td>NaN</td>
      <td>A Nortje</td>
      <td>2023</td>
      <td>A Nortje</td>
      <td>37</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Axar Patel</td>
      <td>9 crore</td>
      <td>900</td>
      <td>0</td>
      <td>NaN</td>
      <td>AR Patel</td>
      <td>2021</td>
      <td>AR Patel</td>
      <td>40</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Axar Patel</td>
      <td>9 crore</td>
      <td>900</td>
      <td>0</td>
      <td>NaN</td>
      <td>AR Patel</td>
      <td>2022</td>
      <td>AR Patel</td>
      <td>182</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```


```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
X = df_merged[['runs_scored']] # Independent variable(s)
y = df_merged['Rs'] # Dependent variable
# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a LinearRegression model
model = LinearRegression()
# Fit the model on the training data
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
X.head()
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
      <th>runs_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
    </tr>
    <tr>
      <th>13</th>
      <td>40</td>
    </tr>
    <tr>
      <th>14</th>
      <td>182</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Assuming df_merged is already defined and contains the necessary columns
X = df_merged[['runs_scored']] # Independent variable(s)
y = df_merged['Rs'] # Dependent variable

# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the model (intercept)
X_train_sm = sm.add_constant(X_train)

# Create a statsmodels OLS regression model
model = sm.OLS(y_train, X_train_sm).fit()

# Get the summary of the model
summary = model.summary()
print(summary)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     Rs   R-squared:                       0.080
    Model:                            OLS   Adj. R-squared:                  0.075
    Method:                 Least Squares   F-statistic:                     15.83
    Date:                Wed, 26 Jun 2024   Prob (F-statistic):           0.000100
    Time:                        00:05:44   Log-Likelihood:                -1379.8
    No. Observations:                 183   AIC:                             2764.
    Df Residuals:                     181   BIC:                             2770.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const         430.8473     46.111      9.344      0.000     339.864     521.831
    runs_scored     0.6895      0.173      3.979      0.000       0.348       1.031
    ==============================================================================
    Omnibus:                       15.690   Durbin-Watson:                   2.100
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               18.057
    Skew:                           0.764   Prob(JB):                     0.000120
    Kurtosis:                       2.823   Cond. No.                         363.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python
from fuzzywuzzy import process

# Convert to DataFrame
df_salary = salary.copy()
df_runs = total_wicket_each_year.copy()

# Function to match names
def match_names(name, names_list):
    match, score = process.extractOne(name, names_list)
    return match if score >= 80 else None  # Use a threshold score of 80

# Create a new column in df_salary with matched names from df_runs
df_salary['Matched_Player'] = df_salary['Player'].apply(lambda x: match_names(x, df_runs['Bowler'].tolist()))

# Merge the DataFrames on the matched names
df_merged = pd.merge(df_salary, df_runs, left_on='Matched_Player', right_on='Bowler')
```


```python
df_merged[df_merged['wicket_confirmation']>10]
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
      <th>Player</th>
      <th>Salary</th>
      <th>Rs</th>
      <th>international</th>
      <th>iconic</th>
      <th>Matched_Player</th>
      <th>Season</th>
      <th>Bowler</th>
      <th>wicket_confirmation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Anrich Nortje</td>
      <td>6.5 crore</td>
      <td>650</td>
      <td>1</td>
      <td>NaN</td>
      <td>A Nortje</td>
      <td>2020/21</td>
      <td>A Nortje</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anrich Nortje</td>
      <td>6.5 crore</td>
      <td>650</td>
      <td>1</td>
      <td>NaN</td>
      <td>A Nortje</td>
      <td>2021</td>
      <td>A Nortje</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Anrich Nortje</td>
      <td>6.5 crore</td>
      <td>650</td>
      <td>1</td>
      <td>NaN</td>
      <td>A Nortje</td>
      <td>2023</td>
      <td>A Nortje</td>
      <td>11</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Axar Patel</td>
      <td>9 crore</td>
      <td>900</td>
      <td>0</td>
      <td>NaN</td>
      <td>AR Patel</td>
      <td>2014</td>
      <td>AR Patel</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Axar Patel</td>
      <td>9 crore</td>
      <td>900</td>
      <td>0</td>
      <td>NaN</td>
      <td>AR Patel</td>
      <td>2015</td>
      <td>AR Patel</td>
      <td>14</td>
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
    </tr>
    <tr>
      <th>589</th>
      <td>T. Natarajan</td>
      <td>3.2 crore</td>
      <td>320</td>
      <td>0</td>
      <td>NaN</td>
      <td>T Natarajan</td>
      <td>2020/21</td>
      <td>T Natarajan</td>
      <td>19</td>
    </tr>
    <tr>
      <th>591</th>
      <td>T. Natarajan</td>
      <td>3.2 crore</td>
      <td>320</td>
      <td>0</td>
      <td>NaN</td>
      <td>T Natarajan</td>
      <td>2022</td>
      <td>T Natarajan</td>
      <td>20</td>
    </tr>
    <tr>
      <th>592</th>
      <td>T. Natarajan</td>
      <td>3.2 crore</td>
      <td>320</td>
      <td>0</td>
      <td>NaN</td>
      <td>T Natarajan</td>
      <td>2023</td>
      <td>T Natarajan</td>
      <td>13</td>
    </tr>
    <tr>
      <th>593</th>
      <td>T. Natarajan</td>
      <td>3.2 crore</td>
      <td>320</td>
      <td>0</td>
      <td>NaN</td>
      <td>T Natarajan</td>
      <td>2024</td>
      <td>T Natarajan</td>
      <td>13</td>
    </tr>
    <tr>
      <th>595</th>
      <td>Umran Malik</td>
      <td>4 crore</td>
      <td>400</td>
      <td>0</td>
      <td>NaN</td>
      <td>Umran Malik</td>
      <td>2022</td>
      <td>Umran Malik</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>227 rows × 9 columns</p>
</div>




```python
#susbsets data for last three years
df_merged = df_merged.loc[df_merged['Season'].isin(['2022'])]
```


```python
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Assuming df_merged is already defined and contains the necessary columns
X = df_merged[['wicket_confirmation']] # Independent variable(s)
y = df_merged['Rs'] # Dependent variable

# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the model (intercept)
X_train_sm = sm.add_constant(X_train)

# Create a statsmodels OLS regression model
model = sm.OLS(y_train, X_train_sm).fit()

# Get the summary of the model
summary = model.summary()
print(summary)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     Rs   R-squared:                       0.074
    Model:                            OLS   Adj. R-squared:                  0.054
    Method:                 Least Squares   F-statistic:                     3.688
    Date:                Wed, 26 Jun 2024   Prob (F-statistic):             0.0610
    Time:                        00:06:13   Log-Likelihood:                -360.96
    No. Observations:                  48   AIC:                             725.9
    Df Residuals:                      46   BIC:                             729.7
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    const                 396.6881     91.270      4.346      0.000     212.971     580.405
    wicket_confirmation    17.6635      9.198      1.920      0.061      -0.851      36.179
    ==============================================================================
    Omnibus:                        6.984   Durbin-Watson:                   2.451
    Prob(Omnibus):                  0.030   Jarque-Bera (JB):                6.309
    Skew:                           0.877   Prob(JB):                       0.0427
    Kurtosis:                       3.274   Cond. No.                         13.8
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python

```
