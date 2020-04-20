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
y = h_{\theta}(x)  = \theta_{0} + \theta_{1}x
$$
$h_{\theta}(x)$: Hypothesis
$y$: Dependent variable
$\theta_{0}$: y-intercept (constant term)
$\theta_{1}$: Coefficient (slope)
$x$: Independent variable

![enter image description here](https://i.imgur.com/oPdsoQH.png =550x)

***- Ordinary least squares*** method: The model minimize the sum of the distances between the observed values and the accordingly modeled value of the linear function:
$$
minimize\sum_{}(y^{(i)} - \hat y^{(i)})^2 
$$

Example:
![enter image description here](https://i.imgur.com/DwaIT6t.png)

The intercept and the slope can be computed directly in this method to formulate the model:
$$
\theta_{1} = \cfrac{\sum_{}(x^{(i)} - \overline{x})(y^{(i)}-\overline{y})}{\sum_{}(x^{(i)}-\overline{x})^2}
$$

$$
\theta_{0} = \overline{y} - \theta_{1}\overline{x}
$$
***- Gradient descent*** method: The model forms the model by iterating and correcting itself. The model will start with random parameters $b_{0}$ and $b_{1}$. For each iteration, it will plug in observed $x_{i}$ and predict $\hat y_{i}$, then compare it with observed $y_{i}$, forming a cost function. The model will then correct itself using cost function

Example:
![enter image description here](https://i.imgur.com/DrCPtXl.png =370x)
>A **cost function** is a measure of how wrong the model is in terms of its ability to estimate the relationship between x and y

There are many types of cost function.  Some popular ones are:
- Sum of Square Error (SSE): $J(\theta_{0},\theta_{1} )=\displaystyle\sum_{i=1}^n(\hat y^{(i)} - y^{(i)})^2$
- Mean of Square Error (MSE): $J(\theta_{0},\theta_{1} )=\cfrac{1}{2m} \displaystyle\sum_{i=1}^m(\hat y^{(i)} - y^{(i)})^2$
$m$: total training samples  (total data points)

[More other cost function](https://towardsdatascience.com/https-medium-com-chayankathuria-regression-why-mean-square-error-a8cad2a1c96f)

The cost function represent the difference (the error) between observed $y$ and predicted $\hat y$. Hence, we need to minimize this cost function in order to formulate the model
Example:
![enter image description here](https://i.imgur.com/UWnCyIZ.jpg =400x)
Formulate the model:
1. Choose starting points for $\theta_{0}$  and $\theta_{1}$, total loop limit, and minimum $error$ limit, and learning rate. The standard for these values are:
	 - $\theta_{0}$ = 0 (intercept)
	 - $\theta_{1}$ = 1 (slope)
	 - $error$ >= 0.001 (minimum value of $error$ for the model to stop)
	 - *total_loop* <= 1000 (maximum loops for the model to stop )
	 - *learning_rate* = 0.1 (the significant level that the model will adjust itself)
	 - the starting point for a model would be: $y = 0 + 1x$
2. Calculate the predicted $\hat{y_{i}}$ by plugging observed $x^{i}$ of each data point into the model
$$
\hat y^{(i)} = 0 + 1x^{(i)}
$$
4. Forming the cost function (in this case we'll use Sum of Squares Error (SSE) cost function) :

$$
cost = \sum(y^{(i)} - \hat y^{(i)} )^2 \\
$$

$$
cost = \sum(y^{(i)} - (\theta_{0} + \theta_{1}x^{(i)}))^2
$$

4. As the model needs to minimize the cost function, next step is to calculate the partial derivative of the $cost$ in accordance to $b_{0}$ or $b_{1}$:

$$
\cfrac{\partial cost}{\partial \theta_{0}} = (\sum(y^{(i)} - (\theta_{0} + \theta_{1}x^{(i)}))^2)'
$$

$$
\cfrac{\partial cost}{\partial \theta_{1}} = (\sum(y^{(i)} - (\theta_{0} + \theta_{1}x^{(i)}))^2)'
$$

5. Calculate the $error$ using the derivative calculated, and adjust it with the *learning_rate* :

$$
error_{\theta_{0}} = \cfrac{\partial cost}{\partial \theta_{0}} * 0.1\\
error_{\theta_{1}} = \cfrac{\partial cost}{\partial \theta_{1}} * 0.1
$$

6. Correct and form a new model using the $error$:

$$
\theta_{0} = \theta_{0} - error_{\theta_{0}}
$$

$$
\theta_{1} = \theta_{1} - error_{\theta_{1}}
$$

7. Repeat the step 2-7 until the $error$ or the *total_loop* reach the limit.

More about OLS and Gradient Descent: [https://www.saedsayad.com/gradient_descent.htm](https://www.saedsayad.com/gradient_descent.htm)
### Multiple Linear Regression

> Multiple linear regression (MLR) is a statistical technique that uses several explanatory variables to predict the outcome of a dependent variable.

*Formula model*:
$$
y = h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \theta_{n}x_{n}
$$

$y$: Dependent variable
$h_{\theta}(x)$: Hypothesis
$\theta_{0}$: y-intercept (constant term)
$\theta_{n}$: Coefficient
$x_{n}$: Explanatory variables (Independent variables)
$x^{(i)}_{j}$: The $i^{th}$ sample of the input samples of the $j$ independent variable 
- i.e: $x^{(2)}_{3}$ is the 2nd input sample of the variable $x_{3}$

$n$: total number of independent variables
$m$: total of input samples

*Variable selection*: Since the model can take in multiple variables, we should select only those variables or predictors which are necessary. These are the methods that for variable selection:
- All-in: Take all the variables into the model. Only do this when you have the prior knowledge or when you are forced to
- Backward Elimination: seeks to remove the variables that do not have a significant effect on the output

	![enter image description here](https://i.imgur.com/kvFQOQV.jpg)
- Forward Selection: begins with an empty model and adds in variables one by one.

	![enter image description here](https://i.imgur.com/6JYnU9k.png)
- Bidirectional Elimination: A combination of Backward Elimination and Forward Selection.

*Formulate the model*
***Gradient Descent*** method: the same Gradient Descent method from Simple Linear Regression can be applied to Multiple Linear Regression. This method seeks to correct itself and minimize the cost function:
- Sum of Square Error (SSE): 
$J(\theta_{0},\theta_{1},..\theta_{n} )=\displaystyle\sum_{i=1}^n(\hat y^{(i)} - y^{(i)})^2$
- Mean of Square Error (MSE): 
$J(\theta_{0},\theta_{1},...\theta_{n} )=\cfrac{1}{2m} \displaystyle\sum_{i=1}^m(\hat y^{(i)} - y^{(i)})^2$
$n$: total predictors (Independent variables)
$m$: total training samples (total data points)

[More other cost functions](https://towardsdatascience.com/https-medium-com-chayankathuria-regression-why-mean-square-error-a8cad2a1c96f)
### Polynomial Linear Regression

> Polynomial regression is a form of regression in which the  relationship between the independent variable *x* and the dependent variable *y* is modeled as an *n*th degree polynomial in *x*

*Formula model*:
$$
y = h_{\theta}(x) = \theta_{0} + \theta_{1}x + \theta_{2}x^2 + ... + \theta_{n}x^n
$$

$y$: Dependent variable
$h_{\theta}(x)$: Hypothesis
$\theta_{0}$: y-intercept (constant term)
$\theta_{n}$: Coefficient (slope)
$x$: Independent variable

Example:
![enter image description here](https://i.imgur.com/QRQNKuz.png)
*Formulate the model*
***Gradient Descent*** method: Just like Linear Regression, Polynomial Linear Regression can use Gradient Descent to formulate the model

### Support Vector Regression
[https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff](https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff)
[https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d](https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d)
[https://www.saedsayad.com/support_vector_machine_reg.htm](https://www.saedsayad.com/support_vector_machine_reg.htm)

## Classification
> Classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known. 

Classification is a type of **supervised learning**.
There are 2 types of Classification:
-   Binomial
-   Multi-Class
Example: 
- Assigning a given email to the “spam” or “non-spam” class

There are 2 types of Classification model:
- Linear Models:
	- Logistic Regression
	- Support Vector Machines
- Nonlinear models:
	- K-nearest Neighbors (KNN)
	- Kernel Support Vector Machines (SVM)
	- Naïve Bayes
	- Decision Tree Classification
	- Random Forest Classification
### Logistic Regression
> Logistic Regression is used to model the probability of a certain class. A binary logistic model has a dependent variable with two possible values, normally labeled "0" and "1".

*Sigmoid Function* (Logistic Function): 

> Sigmoid Function limit the **output** to a range between 0 and 1, making these functions useful in the prediction of probabilities

$$
g(z) = \cfrac{1}{1-e^{-z}}
$$

Example:
![enter image description here](https://i.imgur.com/IPKkckv.png =300x)

*Sigmoid Function* in Regression model:
$$
y = h_{\theta}(x) = g(\theta^Tx) = 
$$

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjEwNTYyMDAyNiwyMTE4MTExODU2LDE0MT
YwODkyOTEsODI2NjIxODU0LDEwODczNDk2MCwtMjY3NTE1NTAw
LDQ1Nzk0MDE5OCw1MTMyNzE0NjksLTk3NDIzNDY0NywtMTc4NT
cxMTE4NywtMTk4MzcwMTg3OCwtMTM3MjgyNDcwNywxOTE5NDEx
OTYxLC01MzgyODA2MiwtMzYyNTE4MjYwLDE5OTU0ODM2NzQsLT
E3NjA5MDM4ODcsLTI4MTMxMzI2NSwtMTQwNDQ5ODIwNCwtMjA5
NDQ1NjAwMF19
-->