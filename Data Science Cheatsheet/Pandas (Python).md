 ## Installation
 ```python
 pip install pandas
 ```
```python
import pandas as pd
 ```

## Load data
[pandas.read_csv()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

Load csv file: `pd.read_csv('filename.csv')`
```python
df_csv = pd.read_csv('filename.csv')
```
Load excel file : `pd.read_excel('filename.csv')`
```python
df_xlsx = pd.read_csv('filename.xlsx')
```
Load .txt file that has delimiter:
```python
df_txt = pd.read_csv('filename.txt',delimiter='\t')
```
From a dictionary:
```python
data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
		'Height': [5.1, 6.2, 5.1, 5.2],
		'Qualification': ['Msc', 'MA', 'Msc', 'Msc']
		}
df.DataFrame(data)
```
Show first `n` lines: `df_csv.head(10)`

Show last`n` lines: `df_csv.tail(10)`

Split data set into chunk to reduce memory:
```python
new_df = pd.DataFrame(columns=df.columns)
for df in pd.read_csv('file.csv', chunksize=1000):
	result = ...
	new_df = pd.concat([new_df,result])

```
## Read data
Read headers: 
```python
df.columns
```
Read specific columns:
```python
df['Name'] #read data index of selected col
df.Name #read data index of selected col
df[['Name','Age','Location']] #read data of indexes of selected cols
df['Name'][start:end] # read 1 data of indexes from start to end of selected col
```
Read specific rows:
```python
df.iloc[1] # read all data of the index of selected row
df.iloc[start:end] # read all data of index from start to end of selected row
```
Read a specific location:
```python
df.iloc[1,2] #row 1, col 2
df.iloc[:, 2: 5] # entire row, col 2-5
```
Iterate over rows: `df.iterrows()`:
```python
for index,row in df.iterrows():
	print(index,row)
	#print(index,row['Name'])
```
[pandas.DataFram.loc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)
Access a group of rows and columns by label(s) or a boolean array.
`df.loc[]`:
```python
df.loc[df['Age'] >= 20] 
# return all indexes that has Age >= 20
df.loc[df['Age'] >=20 & ['Department'] == 'Marketing']
# return all indexes that has Age >=20 and Deparment = Marketing
df.loc[df['Name'].str.contains('John')]
# return all indexes that has Name containing 'John'
df.loc[~df['Name'].str.contains('John')]
# return all indexes that has Name not containing 'John'
df.loc[df['Department'] == 'Mar', 'Department'] = 'Marketing'
# set all values that has 'Mar' into 'Marketing'
```
One-dimensional ndarray `pandas.Series()`
Test if pattern or regex is contained within a string `Series.str.contains(pattern, case=True, flags=0, na=nan, regex=True)`
- case: case sensitive
- flags: regex module
- na: fill in missing values
- regex: use regex or string
```python
s1 = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
s1.str.contains('og', regex=False)
# --> False True False False
s1.str.contains('\d', regex=True)
# --> False False False True
s1.str.contains('dog|house', regex=True)
# --> False True True False
```
Group dataFrame `df.groupby()`
```python
df.groupby(['Department']).sum()
# return a table that has index value of Department column, and sum of all sum-able column 
df['Count'] = 1
df.groupby(['Department']).count()['Count']
# Count the occurance of existence 
```
## Manipulate data
Generate descriptive statistics (count, std, mean, max, ...) of numbers row data. `df.describe()`:

Sort values `df.sort_values(by, axis=0, ascending=True)`:
```python
df.sort_values(['Name'], ascending=True)
df.sort_values(['Name','Age'], ascending=[1,0])
```
Create a new column:
```python
df['New Col'] = value
df['New Col'] = ['value1','value2','value3','value4']
#must match the length of the total index
df['Total'] = df['Col1'] + df['Col2'] - df['Col3']
```
`df.insert(index, columnName, value, allow_duplicates=False)`:
```python
df.insert(2,"Age", [21, 23, 24, 21], True)
```
`df.assign()`:
```python
df.assign(age = [21,23,24,21])
df.assign({'Age': [21, 23]},
	      index=['Andy', 'John'])
```
Drop a column `df.drop(columns)`:
```python
df = df.drop(columns=['Age'])
```
Reset index of the current selection:
```python
df = df.reset_index()
```
Drop duplicate rows: 
```python
df.drop_duplicates(subset='colname', keep={'first','last',False}, inplace=True)
```
## Exporting data
Export csv: `df.to_csv('new.csv', index=Boolean)`
Export xlsx: `df.to_excel('new.xlsx', index=Boolean)`
Export txt: `df.to_csv('new.txt', index=Boolean, sep=['\t'])`
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA2MjQxMzQwNCwtNDQyODA2NjI2XX0=
-->