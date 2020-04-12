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

*Formula model*:
$$
y = b_{0} + b_{1}x
$$
$y$: Dependent variable
$b_{0}$: y-intercept (constant term)
$b_{1}$: Coefficient (slope)
$x$: Independent variable

![enter image description here](https://i.imgur.com/oPdsoQH.png =550x)

***- Ordinary least squares*** method: The model minimize the sum of the distances between the observed values and the accordingly modeled value of the linear function:
$$
minimize\sum_{}(y - \hat{y_{i}})^2 
$$

Example:
![enter image description here](https://i.imgur.com/DwaIT6t.png)

The intercept and the slope can be computed directly in this method to formulate the model:
$$
b_{1} = \cfrac{\sum_{}(x_{i} - \overline{x})(y_{i}-\overline{y})}{\sum_{}(x_{i}-\overline{x})^2}
$$

$$
b_{0} = \overline{y} - b_{1}\overline{x}
$$
***- Gradient descent*** method: The model forms the model by iterating and correcting itself. The model will start with random parameters $b_{0}$ and $b_{1}$. For each iteration, it will plug in observed $x_{i}$ and predict $\hat y_{i}$, then compare it with observed $y_{i}$, forming a cost function. The model will then correct itself using cost function

Example:
![enter image description here](https://i.imgur.com/DrCPtXl.png =370x)
>A **cost function** is a measure of how wrong the model is in terms of its ability to estimate the relationship between x and y

There are many types of cost function. Some popular ones are:
- Sum of Square Error (SSE): $J(b_{0},b_{1} )=\displaystyle\sum_{i=1}^n(\hat y_{i} - y_{i})^2$
- Mean of Square Error (MSE): $J(b_{0},b_{1} )=\cfrac{1}{2n} \displaystyle\sum_{i=1}^n(\hat y_{i} - y_{i})^2$

The cost function represent the difference (the error) between observed $y$ and predicted $\hat y$. Hence, we need to minimize this cost function in order to formulate the model
Example:
![enter image description here](https://i.imgur.com/UWnCyIZ.jpg =400x)
Formulate the model:
1. Choose starting points for $b_{0}$  and $b_{1}$, total loop limit, and minimum $error$ limit, and learning rate. The standard for these values are:
	 - $b_{0}$ = 0 (intercept)
	 - $b_{1}$ = 1 (slope)
	 - $error$ >= 0.001 (minimum value of $error$ for the model to stop)
	 - *total_loop* <= 1000 (maximum loops for the model to stop )
	 - *learning_rate* = 0.1 (the significant level that the model will adjust itself)
	 - the starting point for a model would be: $y = 0 + 1x$
2. Calculate the predicted $\hat{y_{i}}$ by plugging observed $x_{i}$ of each data point into the model
$$
\hat{y_{i}} = 0 + 1x_{i}
$$
4. Forming the cost function (in this case we'll use Sum of Squares Error (SSE) cost function) :

$$
cost = \sum(y_{i} - \hat{y_{i}} )^2 \\
$$

$$
cost = \sum(y_{i} - (b_{0} + b_{1}x_{i}))^2
$$

4. As the model needs to minimize the cost function, next step is to calculate the partial derivative of the $cost$ in accordance to $b_{0}$ or $b_{1}$:

$$
\cfrac{\partial cost}{\partial b_{0}} = (\sum(y_{i} - (b_{0} + b_{1}x_{i}))^2)'
$$

$$
\cfrac{\partial cost}{\partial b_{1}} = (\sum(y_{i} - (b_{0} + b_{1}x_{i}))^2)'
$$

5. Calculate the $error$ using the derivative calculated, and adjust it with the *learning_rate* :

$$
error_{b_{0}} = \cfrac{\partial cost}{\partial b_{0}} * 0.1\\
error_{b_{1}} = \cfrac{\partial cost}{\partial b_{1}} * 0.1
$$

6. Correct and form a new model using the $error$:

$$
b_{0} = b_{0} - error_{b_{0}}
$$

$$
b_{1} = b_{1} - error_{b_{1}}
$$

7. Repeat the step 2-7 until the $error$ or the *total_loop* reach the limit.

More about OLS and Gradient Descent: [https://www.saedsayad.com/gradient_descent.htm](https://www.saedsayad.com/gradient_descent.htm)
### Multiple Linear Regression

> Multiple linear regression (MLR) is a statistical technique that uses several explanatory variables to predict the outcome of a dependent variable.

*Formula model*:
$$
y = b_{0} + b_{1}x_{1} + b_{2}x_{2} + ... + b_{n}x_{n}
$$
$y$: Dependent variable
$b_{0}$: y-intercept (constant term)
$b_{n}$: Coefficient
$x_{n}$: Explanatory variables

*Variable selection*: Since the model can take in multiple variables, we should select only those variables or predictors which are necessary. These are the methods that for variable selection:
- All-in: Take all the variables into the model. Only do this when you have the prior knowledge or when you are forced to
- Backward Elimination: BE seeks to remove the variables that do not have a significant effect on the output
	```mermaid
	graph TD
	A["Choose a Significant Level (i.e SL=5)"]
	A --> B["Fit all the variables into a statiscal model (a model used for </br>compute statiscal data such as variance, P-value, std err,... )"]
	B --> C[Select the variable with the highest P-value]
	C --> D{Is this P-value higher than SL ?}
	```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg5NDAwMzUwNywtMjgxMzEzMjY1LC0xND
A0NDk4MjA0LC0yMDk0NDU2MDAwLC0xMjI1Mjg2NDAwLC0xNjI5
ODM3NTYxLDIyMTAxMjE2MSwxMDQ4OTQ3NDM4LDI4NTU4OTc4Ny
wtMjExOTQzMzUwNCwtMzAwMzQ1NTk0LDE3ODIwNDQ5OTAsNTI3
MDEwNTY0LDMzNTA5OTg4MiwyMTM2NDM3NzMsLTkwMTk0NDAyNC
w3MjgxMDU4MDYsLTE1OTg0MzA3NzksMjA5NDMzNTk0NCw0MzM4
MjA5OTRdfQ==
-->