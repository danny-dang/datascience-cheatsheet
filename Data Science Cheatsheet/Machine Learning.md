## Table of content
- [Data pre-processing](#data-pre-processing)
	- [Deal with missing data](#deal-with-missing-data)
	- [Feature Scaling](#feature-scaling)
	- [Handling outliers](#handling-outliers)
	- [Handling categorical data](#handling-categorical-data)
- [Regression](#regression)
	- [Simple Linear Regression](#simple-linear-regression)
	- [Multiple Linear Regression](#multiple-linear-regression)
	- [Polynomial Linear Regression](#polynomial-linear-regression)
- [Classification](#classification)
	- [Logistic Regression](#logistic-regression)
	- [K-Nearest Neighbors (KNN)](#k-nearest-neighbor-knn)

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
$\theta$: Parameters vector
- i.e: $\theta = \begin{bmatrix}  \theta_{0} \\  \theta_{1} \end{bmatrix}$

$x$: Independent variable vector
- i.e: $x= \begin{bmatrix}  x^{(0)} \\  x^{(1)} \\ ... \\ x^{(m)}\end{bmatrix}$

$m$: total number of input samples

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
- Sum of Square Error (SSE): 
$J(\theta) = J(\theta_{0},\theta_{1} )=\cfrac{1}{2} \displaystyle\sum_{i=1}^n(Cost(h_{\theta}(x^{(i)}),y^{(i)})=\cfrac{1}{2} \displaystyle\sum_{i=1}^n(\hat y^{(i)} - y^{(i)})^2$
- Mean of Square Error (MSE): 
$J(\theta) =J(\theta_{0},\theta_{1} )=\cfrac{1}{2m}\displaystyle\sum_{i=1}^n(Cost(h_{\theta}(x^{(i)}),y^{(i)})=\cfrac{1}{2m} \displaystyle\sum_{i=1}^m(\hat y^{(i)} - y^{(i)})^2$
$m$: total training samples  (total data points)
$\cfrac{1}{2}$ is just for computation convenient 

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
	 - *learning_rate $\alpha$* = 0.1 (the significant level that the model will adjust itself)
	 - the starting point for a model would be: $y = 0 + 1x$
2. Calculate the predicted $\hat{y_{i}}$ by plugging observed $x^{i}$ of each data point into the model
$$
\hat y^{(i)} = 0 + 1x^{(i)}
$$
4. Forming the cost function (in this case we'll use Mean of Squares Error (SSE) cost function) :

$$
\partial J(\theta)= \cfrac{1}{2m}\sum_{i=1}^m(\hat y^{(i)} - y^{(i)} )^2 
$$

$$
\partial J(\theta)=\cfrac{1}{2m} \sum_{i=1}^m((\theta_{0} + \theta_{1}x^{(i)})-y^{(i)})^2
$$

4. As the model needs to minimize the cost function, next step is to calculate the partial derivative of the $cost$ in accordance to $b_{0}$ or $b_{1}$:

$$
\cfrac{\partial J(\theta)}{\partial \theta_{0}} = (\cfrac{1}{2m}\sum_{i=1}^m((\theta_{0} + \theta_{1}x^{(i)})-y^{(i)})^2 )'\\=\cfrac{1}{m}\sum_{i=1}^m(\theta_{0} + \theta_{1}x^{(i)})-y^{(i)})
$$

$$
\cfrac{\partial J(\theta)}{\partial \theta_{1}} = (\cfrac{1}{2m}\sum_{i=1}^m((\theta_{0} + \theta_{1}x^{(i)})-y^{(i)})^2)' \\=\cfrac{1}{m}\sum_{i=1}^m(\theta_{0} + \theta_{1}x^{(i)})-y^{(i)})x^{(i)}
$$

5. Calculate the $error$ using the derivative calculated parameterized by the current value of $\theta_{0}$ and $\theta_{1}$, and adjust it with the *learning_rate* :

$$
error_{\theta_{0}} = \cfrac{\partial J(\theta)}{\partial \theta_{0}} * 0.1\\
error_{\theta_{1}} = \cfrac{\partial J(\theta)}{\partial \theta_{1}} * 0.1
$$

6. Correct and form a new model using the $error$:

$$
\theta_{0} = \theta_{0} - error_{\theta_{0}}
$$

$$
\theta_{1} = \theta_{1} - error_{\theta_{1}}
$$

7. Repeat the step 2-7 until the $error$ or the *total_loop* reach the limit.
*General Formula*:
$$
\text{Repeat until convergence } \{\\
\theta_{0} := \theta_{0} - \alpha \cfrac{1}{m} \sum_{i=1}^m(h_{\theta} (x^{(i)}-y^{(i)} ))\\
\theta_{1} := \theta_{1} -\alpha \cfrac{1}{m}\sum_{i=1}^m(\theta_{0} + \theta_{1}x^{(i)})-y^{(i)})x^{(i)}\\
\}
$$

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
$\theta$: Parameters vector
- i.e: $\theta = \begin{bmatrix}  \theta_{0} \\  \theta_{1} \\ ... \\ \theta_{n} \end{bmatrix}$

$x^{(i)}$: Independent variables vectors of the input sample $i^{th}$
- i.e: $x^{(3)} = \begin{bmatrix}  x^{(3)}_{0} \\  x^{(3)}_{1} \\ ... \\ x^{(3)}_{n} \end{bmatrix}$

$x_{j}$: Input samples vectors of the independent variable $j$
- i.e: $x_{3} = \begin{bmatrix}  x^{(0)}_{3} \\  x^{(1)}_{3} \\ ... \\ x^{(m)}_{3} \end{bmatrix}$

$x^{(i)}_{j}$: The $i^{th}$ sample of the input samples of the $j$ independent variable 
- i.e: $x^{(2)}_{3}$ is the 2nd input sample of the variable vector $x_{3}$

$x$: Matrix of independent variables and input samples
- $x =\begin{bmatrix}  x^{(0)}_{0} & x^{(1)}_{0} & ... & x^{(m)}_{0}  \\  x^{(0)}_{1} & x^{(1)}_{1} & ... & x^{(m)}_{1}\\ ... & ... & ... & ... \\ x^{(0)}_{n} & x^{(1)}_{n} & ... & x^{(m)}_{n}\end{bmatrix}$

$n$: total number of independent variables vectors
$m$: total number  of input samples

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
$J(\theta) =J(\theta_{0},\theta_{1},..\theta_{n} )=\cfrac{1}{2}\displaystyle\sum_{i=1}^n(Cost(h_{\theta}(x^{(i)}),y^{(i)})=\cfrac{1}{2}\displaystyle\sum_{i=1}^n(\hat y^{(i)} - y^{(i)})^2$
- Mean of Square Error (MSE): 
$J(\theta)=J(\theta_{0},\theta_{1},...\theta_{n}) =\cfrac{1}{2m}\displaystyle\sum_{i=1}^n(Cost(h_{\theta}(x^{(i)}),y^{(i)}))=\cfrac{1}{2m} \displaystyle\sum_{i=1}^m(\hat y^{(i)} - y^{(i)})^2$
$n$: total predictors (Independent variables)
$m$: total training samples (total data points)
$\cfrac{1}{2}$ is just for computation convenient 
*General Formula*:
$$
\text{Repeat until convergence }\{\\
\theta_{j} := \theta_{j} -\alpha\cfrac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j}\\
...\\
\}
$$

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
$\theta$: Parameters vector
$x$: Independent variable vector

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

***Sigmoid Function*** (Logistic Function): 

> Sigmoid Function limit the **output** to a range between 0 and 1, making these functions useful in the prediction of probabilities

$$
g(z) = \cfrac{1}{1-e^{-z}}
$$

Example:
![enter image description here](https://i.imgur.com/IPKkckv.png =300x)

*Sigmoid Function* in Regression model:
$$
y = h_{\theta}(x) = g(\theta^Tx) = \cfrac{1}{1-e^{-\theta^Tx}} \\   \\
$$

$y$: Dependent variable
$h_{\theta}(x)$: Hypothesis
$\theta$: Parameters vector
$\theta^T$: Tranpose of the parameters vector
$x$: Matrix of Independent variables and input samples
$\theta^Tx$: $\begin{bmatrix}  \theta_{0} &  \theta_{1} & ... & \theta_{n} \end{bmatrix} \begin{bmatrix}  x_{0} \\  x_{1} \\ ... \\ x_{n} \end{bmatrix}
 = \theta_{0}x_{0} + \theta_{1}x_{1} + ... + \theta_{n}x_{n}$

*Hypothesis represents **probability***:

$h_{\theta}(x)$: The probability that $y = 1$ on input $x$
$h_{\theta}(x) = P(y=1|x;\theta)$: Probability that y =1, with input variable x, and parameter $\theta$
- i.e: 
Tell if a mail is a spam, in which "1" is spam, "0" is not spam
$h_{\theta}(x) = 0.25$ : 25% chance of mail is spam

*Hypothesis  represents **Decision Boundary***: separates the data-points into regions, which is the classes where data points belong.

$h_{\theta}(x)  \geq 0.5 \implies y =1$
$h_{\theta}(x) < 0.5 \implies y =0$
__
$h_{\theta}(x)  \geq 0.5 \iff g(\theta^Tx)   \geq 0.5$
$h_{\theta}(x) < 0.5 \iff g(\theta^Tx)  < 0.5$
__
$g(\theta^Tx)   \geq 0.5\iff\theta^Tx\geq0$
$g(\theta^Tx)  < 0.5\iff\theta^Tx<0$
__
$\theta_{0}x_{0} + \theta_{1}x_{1} + ... + \theta_{n}x_{n}\geq0 \implies y =1$
$\theta_{0}x_{0} + \theta_{1}x_{1} + ... + \theta_{n}x_{n}<0 \implies y =0$

$\theta_{0}x_{0} + \theta_{1}x_{1} + ... + \theta_{n}x_{n}=0$ is the boundary line that separate 2 classes
Example:
![enter image description here](https://i.imgur.com/lqW86d4.png =400x)

*Formulate the model:*

**Gradient Descent** method:
Logistic Regression Cost:
$$
Cost(h_{\theta}(x),y) = \begin{cases}-log(h_{\theta}(x)) & \text{if } y=1  \\-log(1-h_{\theta}(x)) & \text{if } y=0 \end{cases}
$$

$$
Cost(h_{\theta}(x),y) = - [ylog(h_{\theta}(x)) + (1-y)log(1-h_{\theta}(x))]
$$
Cost Function:
$$
J(\theta)=\cfrac{1}{m}\sum_{i=1}^mCost(h_{\theta}(x^{(i)}),y^{(i)}) \\
= \cfrac{1}{m}\sum_{i=1}^m [-y^{(i)}log(h_{\theta}(x^{(i)})) - (1-y^{(i)})log(1-h_{\theta}(x^{(i)}))]
$$
Cost Function Derivative: 
$$
\cfrac{\partial J(\theta)}{\partial \theta_{0}} =\cfrac{1}{m}\sum_{i=1}^m [-y^{(i)}log(h_{\theta}(x^{(i)})) - (1-y^{(i)})log(1-h_{\theta}(x^{(i)}))]' \\
= \cfrac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j}
$$
Self correcting loop:
$$
\text{Repeat until convergence }\{\\
\theta_{j} := \theta_{j} -\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j}\\
...\\
\}
$$

## K-Nearest Neighbor (KNN)

> The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

KNN uses the current data points to predict the class of a new data points based on $k$ - the number of the nearest existing data points (neighbors).

KNN uses the current existing data points to predict rather than using a model.

*Classify a new data point*:
1. Choose number of $k$ neighbors (i.e: k=3)
2. Calculate the distance to the nearest neighbors:
$$d(P;Q) = \sqrt{ \sum_{i=1}^{n}(q_{i} - p_{i})}$$

	$P = (p_{1},p_{2},...,p_{n}) \in \Reals^n$
	$Q = (q_{1},q_{2},...,q_{n})\in \Reals^n$
	
	$P$ is the new data point, and $Q$ is the existing data point
3. Select the top k neighbors that has the lowest $d$:

	![enter image description here](https://i.imgur.com/i0NPF6F.jpg =350x)
	
4. Based on the highest number of neighbors, the new data point will belongs to that class. 
	- i.e: $k=3$ ,new $P$ near 2 data points of class A, and near 1 data point of class B, new $P$ will belongs to class A.
	- If $k$ is even (i.e $k=4$), and the number of nearest neighbors is tie for each class (i.e 2 neighbors for A, 2 neighbors for B), you can choose either to random, or to use the nearest neighbor to break the tie.
	
*Weighted KNN*:

*Decision Boundary:* the Decision Boundary is formed where the distances between 2 points of 2 classes are equal.
![enter image description here](https://i.imgur.com/zWF2bBQ.png =300x)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk1NjEzMzI2MywxOTE2NzcyOTE2LDEyMz
I5OTU4NTQsMTMwMTEzNDc4NiwtMjUzMDA5NDAsLTIwMTUwMTg3
OCwtMTIzMDA5MzQ1MiwtNDI2ODYxMTY2LDEwNjMwODc3ODYsMT
IzMTMzNzc3NCwxNzU4MjU3MTYsMTU1NTkzNTA3MSwtMzc0MzQz
MjU3LC0xNzYyODEwNjI0LC0zNjU3NzQxODgsMTM5NDAyMzc3OC
wxNjY4MjczNTg2LDE4ODI1NDA0NzEsLTg5NjgxNzU3NiwtMTIz
NTYxMzk0NF19
-->