## Data pre-processing
Sample data set: *Survey*
```
    Location    Age    AnnualSalary      Opinion  
    ________    ___    ____________    ___________
    'US'        40        72000        'not liked'
    'Asia'      25        48000        'very liked'
    'Africa'    30        54000        'not liked'
    'Asia'      35        61000        'not liked'
    'Africa'    44        63777        'liked'    
    'US'        33        58000        'very liked'
    'Asia'      37        52000        'not liked'
    'US'        40        79000        'liked'    
    'Africa'    55        83000        'not liked'
    'US'        35        67000        'liked'   
```       
### 1. Deal with missing data

**Remove missing data:**
`data_set = rmmissing(table_data,dim,Name,Value)`
```matlab
survey_data = rmmissing(survey_data,1,'MinNumMissing',3)
% remove the row with more than 3 missing data
```
**Calculate mean, mode, median and fill in missing data:**
`fillmissing(A,'constant',v)`
```matlab
mean_Age = mean(survey_data.Age,'omitnan') 
% omitnan to ignore NaN values in calculation
student_data.Age = fillmissing(survey_data.Age,'constant',mean_Age)
```
Other fill missing method: `fillmissing(A,method)`

### 2. Feature Scaling

**Normalization:**
```matlab
survey_data.Age = normalize(survey_data.Age,'range')
```
**Standardization:**
```matlab
survey_data.Age = zscore(survey_data.Age)
```
### 3. Handling Outliers

**Remove outliers:**

- Return a logical array of outliers: `isoutlier(A,method)`
```matlab
survey_data = survey_data(~isoutlier(survey_data.Age,'median'),:)
```
**Fill in outliers**

Fill in method: `filloutliers(A,fillmethod,findmethod)`
```matlab
survey_data.Age = filloutliers(survey_data.Age,'center','median')
```
### 4. Handling Categorical Data
Matlab doesn't have built-in function, so we need to create our own:

**Label Encoding**
```matlab
function new_variable = labelEncoding(variable,values_set,numbers) 
	[rows,col] = size(variable);
	new_variable = zeros(rows,1);

	for i=1:length(values_set)
	    indices = ismember(variable,values_set{i});
	    new_variable(indices) = numbers(i);
	end 
end 
```
```matlab
survey_data.Opinion = labelEncoding(survey_data.Opinion,{'not liked','liked','very liked'},[0 1 2])
```
**One Hot Encoding**
```matlab
function data = oneHotEncoding(data,variable)
	unique_values = unique(variable);
	 
	for i=1:length(unique_values)
	    dummy_variable(:,i) = double(ismember(variable,unique_values{i})) ;
	end 

	T = table;
	[rows, col] = size(dummy_variable);

	for i=1:col
	    T1 = table(dummy_variable(:,i));
	    T1.Properties.VariableNames = unique_values(i);
	    T = [T T1];
	end
	
    data = [data T]; 
end 
```
```matlab
survey_data.Opinion = onHotEncoding(survey_data,survey_data.Location)
```
Avoid dummy variable trap:
```matlab
survey_data = survey_data(:,1:end-1)
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzM0MzM2OTU3LDY5NDUyMzI2NiwtMTQyNj
I4Nzg2MiwtMzExNDAwODY5LDE4NzQ2NjQ4MTUsLTU0NTEwOTk5
NiwtODI0ODUyNDk2LDY0ODUzMjI3LC0xNTQ1OTc2MjY0LC0xNT
E5MzI1Mzg5LC0xNjk3Njk5ODI2LDgwMzI5OTA0MSw3OTIyMjg5
OTcsMTA3NjY5MTYwOCwtMjc3MzM0OTAwLDQwODE1OTgwMCwzMD
g2NDIwNjJdfQ==
-->