 ## Installation
 ```python
 pip install numpy
 ```
```python
import numpy as np
 ```
 ## Array creation
 Note: `axis=0` y-axis; `axis=1` x-axis; `axis=2` z-axis
 Create a numpy array `np.array([])`:
 [numpy.array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)
```python
arr1 = np.array([1,2,3])
print(arr)
#--> [1,2,3]

arr2 = np.array([[1,2,3],[4,5,6])
print(arr)
#--> [[1,2,3],
#-->   4,5,6]]
```
 Get array dimension `arr.ndim`:
```python
print(arr2.ndim)
#--> 2
```
 Get array shape`arr.shape`:
```python
print(arr2.shape)
#--> (2,3) (row, col)
```
Get data type `arr.dtype`

Get size (bytes) of each item in array `arr.itemsize`

Get total size of an array `arr.nbytes`

Create an array with full of a number `np.full(shape,num)`:
```python
arr3 = np.full((2,2),99)
#--> [[99,99],[99,99]]
```
Create an array with random decimal numbers `np.random.rand(shape)`

Create an array with random integer numbers `np.random.randint(lowest,highest=None,size=None)`

Create an identity matrix `np.identity(size)`
Repeat elements of an array `numpy.repeat(arr,repeats,axis=None)`

```python
x = np.array([[1,2],[3,4]])
np.repeat(x, [1, 2], axis=0)
#-->array([[1, 2],
#          [3, 4],
#          [3, 4]])
```
## Manipulate array
```python
arr = np.array([[1,2,3],[4,5,6])
arr2 = np.array([[[1,2],[3,4]],
                 [[5,6],[7,8]]])
```
Basic select:
```python
print(arr[1,2])
#--> 6
arr[0,:]
#--> [1,2,3]
arr[:,0]
#--> [1,4]
```
Using `:`
```python
#arr[startidx:endidx:step]
arr[0:2:1,1]
#--> [4,6]
arr[0:-1:1,1]
#--> [4,6]
arr2[:,0,:]
#--> [[1,2],[5,6]]
```
Select a list:
```python
arr = np.array([1,2,3,4,5,6])
arr[[1,4]] #--> [2,5]
```
Using `> <`:
```python
arr = np.array([1,2,3,4,5,6])
z > 3 #--> [False, False, False, True, True, True]
z[z > 3] #--> [4,5,6]
```
Change value of items
```python
arr[0,2] = 9 #one item
arr[:,1] = 8 #entire col
arr[:,1] = [10, 11] #entire col with specifix number
arr2[:,0,:] = [[1,1],[2,2]] #need to be the same dimension to the selected
```
Reshape an array `np.reshape(a,newshape)` or `arr.reshape(newshape)`:
```python
arr = np.array([[1,2,3,4],[5,6,7,8]])
arr.reshape((4,2))
#--> [[1,2],
#--> [[3,4],
#--> [[5,6],
#-->  [7.8]]

#The new shape should be compatible with the number of items 
#in the old shape
arr.reshape((2,3)) #--> err
```
Vertically vectors stacking `np.vstack([v1,v2])`:
```python
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
np.vstack([v1,v2])
#-->[[1,2,3],[4,5,6]]
```
Horizontally vectors stacking `np.hstack([v1,v2])`:
Test whether any array element along a given axis evaluates to True.
`np.any(arr, axis=None)`

Returns a ndarray object containing evenly spaced values within the given range `np.arrange(start, stop, step, dtype)`
```python
np.arrange(start=1,stop=8, step=2)
#--> [1,3,5,7]
```
Returns a ndarray object containing evenly spaced numbers over a specified interval `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`:
```python
np.linspace(2.0, 3.0, num=5)
#--> array([2.0, 2.25, 2.5, 2.75, 3.0])
np.linspace(2.0, 3.0, num=5, endpoint=False)
#--> array([2.0,  2.2,  2.4,  2.6,  2.8])
np.linspace(2.0, 3.0, num=5, retstep=True)
#--> (array([2.0,  2.25,  2.5 ,  2.75,  3.0]), 0.25)
```
Return elements chosen from `x` or `y` depending on `condition`. where True, yield `x`, False yield `y`.
`numpy.where(condition,x,y)`
```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.where(a < 5, a, 10*a)
#--> array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
```
Return coordinate matrices from coordinate vectors. (for making linear grid) `np.meshgrid(x,y)` :
```python
xvalues = np.array([0, 1, 2, 3, 4]);
yvalues = np.array([0, 1, 2, 3, 4]);
xx, yy = np.meshgrid(xvalues, yvalues)
# xx
[[0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]]
 # yy
[[0 0 0 0 0]
 [1 1 1 1 1]
 [2 2 2 2 2]
 [3 3 3 3 3]
 [4 4 4 4 4]]
```
## Math with Numpy
[Numpy Math](https://docs.scipy.org/doc/numpy/reference/routines.math.html)
Expression `+ - / * **`, `sin(arr) cos(arr) tan(arr)`:
```python
arr = np.array([1,2,3,4])
arr + 2
#--> [3,4,5,6]
```

Find sum `np.sum(arr,axis=None)`:
```python
np.sum(arr1,axis=0)
#--> [2,4]
```
Find product `np.prod(arr,axis=None)`

## Linear Algebra
[Linear Algebra docs](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)

Multiply 2 matrices `np.matmul(arr1,arr2)`:
```python
arr1 = np.array([[1,2,3],
                 [4,5,6]])
arr2 = np.array([[1,2],
                 [3,4],
                 [5,6])
np.matmul(arr1,arr2)
```
Create an identity matrix `np.identity(size)`
Calculate determinant (định thức) `np.linalg.det(arr)`

## Statistics 
[Numpy Statistics](https://docs.scipy.org/doc/numpy/reference/routines.statistics.html)
Return the minimum of an array or minimum along an axis `np.amin(arr,axis=None)`:
```python
arr1 = np.array([[0, 1],
                 [2, 3]])
np.amin(arr1) #--> 0
np.amin(arr1, axis=0) #--> [0,1]
```
Same to `np.amax(arr,axis=None)`
Find mean `np.mean(arr,axis=None)`
Find variance `np.var(arr,axis=None)`
Find average `np.average(arr,axis=None)`
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg5MzE4MjMwMiwxNDcyMDU0MjZdfQ==
-->