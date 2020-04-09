## Table of content
- [Data pre-processing](#data-pre-processing)
	- [Deal with missing data](#deal-with-missing-data)
	- [Feature Scaling](#feature-scaling)
	- [Handling outliers](#handling-outliers)
	- [Handling categorical data](#handling-categorical-data)
- [Regression](#regression)
	- [Simple Linear Regression](#simple-linear-regression)
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
'Africa'    55        83000        'not liked' 
```    
### Deal with missing data
Remove missing data

Replace missing data 
-	Calculate mean, median, mod (exclude missing number in calculation)
-	Replace with the most frequent value
-	Using algorithms

### Feature Scaling

> When there are big differences in the range of values of different
> variables, these values need to be standardize into a fixed range.

*i.e: Age and AnnualSalary*

Normalization:
- *Rescaling the range of features to scale the range in [0, 1] or [−1, 1].*
- Formula:
	$$
	x_{new}  =\cfrac{x - min(x)}{max(x) - min(x)}
	$$

Standardization 
- *Rescale data to have a mean of 0 and a standard deviation of 1.*
- Formula:
	$$
	x_{new}  =\cfrac{x - \overline{x}}{\sigma}
	$$
### Handling Outliers

> An outlier is a data point that differs significantly from other
> observations.

![enter image description here](https://i.imgur.com/cQUTmj8.gif)

Different library will have different strategies to handle outliers aiming at *removing the outliers* or *filling in the outliers*.

### Handling Categorical Data

> Categorical variables represent types of data which may be divided
> into groups.

*i.e: Location, Opinion*

**Label Encoding**

> Label encoding converts the data into numeric forms

i.e:
```
not liked -> 0
liked -> 1
very liked ->2
```
``` 
Location    Age    AnnualSalary      Opinion  
________    ___    ____________    ___________
'US'        40        72000             0
'Asia'      25        48000             2
'Africa'    30        54000             0
'Asia'      35        61000             0
'Africa'    44        63777             1    
'US'        33        58000             2 
'Asia'      37        52000             0
'Africa'    55        83000             0 
```
**One Hot Encoding**
> One Hot Encoding converts the data which has no relationship into dummy variables with the value of 0 and 1.

i.e 
``` 
Age    AnnualSalary      Opinion	  Africa  Asia	  US
___    ____________    ___________    ______  ____   ____
40        72000        		0	        0      0	   1
25        48000        		2 	        0	   1	   0 
30        54000        		0	        1      0	   0 
35        61000        		0	        0      1	   0
44        63777        		1           1      0	   0 
33        58000             2	        0      0	   1
37        52000             0	        0      1	   0
55        83000             0           1      0	   0
``` 
*Dummy variable trap:* When using One Hot Encoding, there are attributes which are highly correlated, meaning that variables can be predicted from the others.
*i.e: The data in the US col can be predicted from the 2 cols Africa and Asia. So removing one of the 3 cols is essential to avoid dummy variable trap*
``` 
Age    AnnualSalary      Opinion	  Africa  Asia	
___    ____________    ___________    ______  ____  
40        72000        		0	        0      0	   
25        48000        		2 	        0	   1	    
30        54000        		0	        1      0	    
35        61000        		0	        0      1	   
44        63777        		1           1      0	    
33        58000             2	        0      0	   
37        52000             0	        0      1	   
55        83000             0           1      0	   
``` 
## Regression

### Simple Linear Regression

> Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables

Formula model:
$$
y = b_{0} + b_{1}x
$$
$y$: Dependent variable
$b_{0}$: y-intercept (constant term)
$b_{1}$: Coefficient (slope)
$x$: Independent variable

![enter image description here](https://i.imgur.com/oPdsoQH.png =550x)

***Ordinary least squares*** method: The model minimize the sum of the distances between the observed values and the accordingly modeled value of the linear function:
$$
min(\sum_{}(y - \hat{y_{i}})^2 )
$$

Example:
![enter image description here](https://i.imgur.com/DwaIT6t.png)

Formulate the model:
$$
b_{1} = \cfrac{\sum_{}(x_{i} - \overline{x})(y_{i}-\overline{y})}{\sum_{}(x_{i}-\overline{x})}
$$

$$
b_{0} = \overline{y} - b_{1}\overline{x}
$$
***Gradient descent*** method: The model seeks to optimize and correct itself, starting from a random point of  $b_{0}$  and $b_{1}$. For each time $x$ (observed $x$) is plugged into the model, the model will return $\hat{y}$, (predicted $\hat{y}$) this $\hat{y}$ will be compared to $y$ (observed $y$) to find $error$, forming a cost function to correct $b_{0}$  and $b_{1}$. The step will be repeated until the model reaches a limit of total step or $error$ reaches minimum value.

Example:
![enter image description here](https://i.imgur.com/DrCPtXl.png =350x)

Formulate the model:
1. Choose starting points for $b_{0}$  and $b_{1}$, total loop limit, and minimum $error$ limit, and learning rate, the standard for these values are:
	 - $b_{0}$ = 0 (intercept)
	 - $b_{1}$ = 1 (slope)
	 - $error$ >= 0.001 (minimum value of $error$ for the model to stop)
	 - *total_loop* <= 1000 (maximum loops for the model to stop )
	 - *learning_rate* = 0.1 (the significant level that the model will adjust itself)
	 - the starting point for a model would be: $y = 0 + 1x$
2. Calculate the predicted $\hat{y_{i}}$ by plugging observed $x_{i}$ of each data point into the model
3. Forming the cost function that's equaled the sum of the squared of the difference between observed $y_{i}$ and predicted $\hat{y_{i}}$:

$$
cost = \sum(y_{i} - \hat{y_{i}} )^2 \\
$$

$$
cost = \sum(y_{i} - (b_{0} + b_{1}x_{i}))^2
$$

4. As the model needs to minimize the cost function, next step is to calculate the partial derivative of the $cost$ in accordance to $b_{0}$ or $b_{1}$:

$$
\cfrac{\partial}{\partial b_{0}} = (\sum(y_{i} - (b_{0} + b_{1}x_{i}))^2)'
$$

$$
\cfrac{\partial}{\partial b_{1}} = (\sum(y_{i} - (b_{0} + b_{1}x_{i}))^2)'
$$

5. Calculate the $error$ using the derivative calculated, and adjust it with the *learning_rate* :

$$
error_{b_{0}} = \cfrac{\partial}{\partial b_{0}} * 0.1\\
error_{b_{1}} = \cfrac{\partial}{\partial b_{1}} * 0.1
$$

6. Correct and form a new model using the $error$:

$$
b_{0} = b_{0} - error_{b_{0}}
$$

$$
b_{1} = b_{1} - error_{b_{1}}
$$

7. Repeat the step 2-7 until the $error$ or the *total_loop* reach the limit.
### Multiple Linear Regression

> Multiple linear regression (MLR) is a statistical technique that uses several explanatory variables to predict the outcome of a response variable.

Formula model:
$$
y = b_{0} + b_{1}x_{1} + b_{2}x_{2} + ... + b_{n}x_{n}
$$
$y$: Dependent variable
$b_{0}$: y-intercept (constant term)
$b_{n}$: Coefficient
$x_{n}$: Explanatory variables
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA5NDMzNTk0NCw0MzM4MjA5OTQsMTgzOT
M0OTc5NSwxNzI3OTExODEyLDIwNDg5OTMzODgsLTExMzI5Njg3
OTYsLTEwNzc5NjQ0ODIsMTIzOTU0NTAwMywtMTk3MTg0MDczNS
w3OTk4NzEwMzMsMTQ3MjIzNDE1MCw2NTQ4OTQ4NTksLTEwODU4
Mjc0NjksNDk2NTc4MzE1LDUwMzgxMTAwOCwtMTU5MjUyODEzNi
wtMTM0MjA4NzE1LC0xNTI5ODk4NTE4LC01MzU2OTMxODYsMzA5
ODM3NzI3XX0=
-->