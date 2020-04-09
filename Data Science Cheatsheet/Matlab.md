## Table of contents
1. [Defining variables](#defining-variables)
2. [Manipulate string](#manipulate-string)
3. [Manipulate matrices](#manipulate-matrices)
4. [Math](#math)
5. [Statistics](#statistics)
6. [Algebra](#algebra)
7. [Matlab Datatypes](#matlab-datatypes)
8. [Programming](#programming)

## Defining variables
**Define a number:**
```matlab
x = 5
x = true % -> 1
x = ~true % -> 0
```
**Define a vector:**
```matlab
v = [1,2,3]
v2 = [ 1 2 3 ] % row vector
v3 = [ 1; 2; 3; ] %column vector
v4 = 1:3 % -> [1 2 3]
v5 = 1:2:10 % start:step:end 
%-> [1 3 5 7 9]
```
**Define a matrix:**
```matlab
A = [ 1 2; 3 4 ]
%-> [1  2;
%    3  4]
B = [ 1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
```
**Define a symbolic variable:**
```matlab
a = 1/3
%-> 0.3333
a = sym(1/3)
%-> 1/3
syms x
y = sym(2x+1)
%-> 2x+1
```
**Show all the variables defined:** `whos`

**Clear variables:** `clear var_name`, `clear all`

**Load data:** `load('file.mat')`

**Read csv file:** `csvread('file.csv')`

**Input, display variable:** 
```matlab
x = input('Please enter something')
disp(x)
y = menu('Select', 'Option 1', 'Option 2')
a = "highest number"
b = 9.999
fprintf("The %s is %.4f", a, b)
%-> The highest number is 9.9990
```
[↑ Back to top](#table-of-contents)
## Manipulate string
**Read file:** `fileread('./file.txt')`

**Split string or text lines:** `splitlines(text)`

**Replace substring:** `replace(text,{',',';'},' ')`

**Trim spaces:** `strip(text)`

**Split words into string array:** `split(text)`

**Join string array in to 1 string:** `join(text)`

[↑ Back to top](#table-of-contents)
## Manipulate matrices
**Access element in a matrix:**
```matlab
A(1,2)
%-> 3
A(:,2)
%-> [3;
%    4]
B(2:3,2:3)
%-> [6  7;
%   10 11]
B(1:2:end,:)
%-> [1 2  3  4;
%   9 10 11 12]
```
**Return size of a matrix:** `size(A)`

**Return dimension of a matrix that has the highest value:** `length(A)`

**Check if element(s) in a matrix:** `ismember(x,A)`
```matlab
A = [1 2 3; 5 6 0]
ismember(5,A) % -> 1
ismember([1 4 9],A) % -> [1 0 0]
ismember([5 6 0],A) % -> 1
```
**Shift elements in a matrix**: `circshift(A,[x_shiftLen,y_shiftLen])`
```matlab
A = [1 2 3; 4 5 6; 7 8 9]
cirshift(A,[1,1])
%-> 9  7  8
%   3  1  2
%   6  4  5
```
**Concat 2 matrices:** 
```matlab
A = [1 2 3; 4 5 6; 7 8 9]
B = [1 2 3]
[A; B]
%-> 1 2 3
%   4 5 6
%   7 8 9
%   1 2 3
```
**Relational operations, return `0 ` `1`:**
```matlab
A > B % gt(A,B)
A < B % lt(A,B)
A >= B % ge(A,B)
A <= B % le(A,B)
A ~= B % ne(A,B)
A == B
```
**Find indexes of non-zero value:** `find(A)`

**Sort matrix:**  `sort(A, dimension, direction)`

**Find the frequency of the values in a vector:** `tabulate(v)`

[↑ Back to top](#table-of-contents)

## Math
**Basic expression:**
```matlab
a+b
a-b
a*b
a/b
a^2
```
**Calculate sin, cos, tan:**
```matlab
cos(x)
sin(x)
tan(y)
cos(A)
```
**Calculate `e`, `ln`, and `log`:**
```matlab
exp(1) % e
exp(2) % e^2
log2(8)
log(1) % ln(1)
```
**Calculate differentiation (derivative) (đạo hàm):** `diff(a,n,dim)`
```matlab
syms x
diff(2x)
%-> 2
diff(2*x^2,2) % diff(diff(2*x^2))
%-> 4
```
**Calculate integration (tích phân):** 
```matlab
% definite integral (antiderivative) 
int((2*x-1)^2)
%-> ((2*x - 1)^3)/6

% indefinite integral
int((2*x-1)^2,[1 10])
%-> 1143
```
**Symbolic function:**
```matlab
syms x y
f(x,y) = 2*x + 3*y +2
f(2,3)
%-> 15
```
**Solve equations** `solve(f)`:
```matlab
syms f x y
f = 2*x^2 + 3*x + 2 == 1
solve(f)
%-> -1
%   -1/2
f = 2*x^2*y^2 + 3*x*y + 2 == 1
solve(f)
%-> -1/y
%   -1/(2*y)
solve(f,y)
f1 = 3*x + 2*y == 1
f2 = 2*y + 3*z == 2
f3 = 3*z == 3
[x y z] = solve(f1,f2,f3)
```
**Substitute symbolic variable** `subs(f,old,new)`:
```matlab
f = 3*x + 2*y == 1
subs(f,x,1)
%-> 2*y + 3 == 1
```
**Calculate product:** `prod(A)`

**Find greatest common divisor (GCD):** `gcd(x,y)`

**Least common multiple (LCM):** `lcm(x,y)`

**Check if is prime number:** `isprime(x)`,  `isprime(A)`

**Return all the permutations:** `perms([1 2 3])`

**Set operations:**
```matlab
intersect(x,y)
union(x,y)
setdiff(x,y)
```
**Generate random values:**
```matlab
rand(5,2) % rand(row,col) random between 0 and 1
randn(5,2) % randn(row,col) random negative -1 and 0
randi(10,3,2) % randi(max,row,col) random integer from 1 to max

```
**Discretize values (turn values into discrete parts):** 
```matlab
data = [1 3 8 5 9 7]
edges = [0 2 4 6 8]
A = discretize(data,edges)
% --> [1 2 4 3 NaN 4]
% bin 1: 0-1.99; bin 2: 2-3.99; bin 3: 4-5.99; bin 4: 6-8
```
**Find unique numbers** `unique(A)`

[↑ Back to top](#table-of-contents)

## Statistics
```matlab
A = [1 3 5; 3 6 9; 7 8 2]
```
**Find min:** `min(A)`
```matlab
min([1,2,3,4]) % 1 (min of the vector)
min(A) % 1 3 2 (min of every column)
min(A,[],1) % min of every column
min(A,[],2) % min of every row
[A, I] = min(A,[],1) % return a min matrice and a matrice of its indexes
```
**Find max:** `max(A)`

**Find mean:** `mean(A)`
```matlab
mean([1 2 3]) % -> 2 (mean of the vector)
mean(A) % -> [3.6667 5.6667 5.3333] (mean of every col)
mean(A,2) % mean of every row
```
**Find sum**: `sum(A)`

**Find variance:** `var(A)`:
```matlab
var([1 2 3]) % 1 (variance of the vector)
var(A) % (variance of every col)
var(A,0,2) % (variance of every row)
```
**Find standard division** `std(A)`

**Find median** `median(A)`

**Find mod** `mod(A)`

**Find percentile** `prctile(A,nth)` 
```matlab
prctile(A,25) % 25th percentile of every col
prctile(A,[25,35,45] % percentile of the according col
prctile(A,25,2) % 25th percentile of every row
```
**Signum and Absolute value:** `sign(x)`, `abs(x)`

[↑ Back to top](#table-of-contents)
## Algebra
**Add, subtract, multiply, divide vectors and matrices:**
```matlab
A = [1 2; 3 4]
B = [3 4; 5 6]
A+B
A-B
A*B
A\B % = A * inv(B)
```
**Multiply every single element of one to another matrix:**
```matlab
A.*B
%-> [3  8;
%-> 15 24]
A.\B
```
**Calculate products:**
```matlab
dot(x,y)
cross(x,y)
outer(x,y)
inner(x,y)
```
**Transpose a matrix or vector:**
```matlab
A = A'
%-> [1  3;
%-> 2  4]
```
**Generate an identity matrix:** `eye(3)`

**Find the inverse matrix:** `inv(A)` `A^-1`

**Find the determinant:** `det(A)`

**Return diagonal elements:** `diag(A)`

[↑ Back to top](#table-of-contents)

## Matlab Datatypes
### Cell array

> Cell array can contains any types of data in each cell.

**Create a cell array:**
```matlab
C = {1 2 'text';[1 2; 3 4] {1 2} 3}
C2 = cell(4,2) % create empty cell of 4,2
```
**Visualize cell:** `cellplot(C)`

**Display cell:** `celldisp(C)`

**Access data in a cell:**
```matlab
C{1,1}
% -> 1
C{2,1}
% ->1 2
%   3 4
C{2,1}(1,2)
% -> 2
C{2,2}{1}
% -> 1
```
**Replace data:**
```matlab
C{2,2} = 9
C{:,1:2} = [] % Remove col 1-2
```
### Tables

> A table is like a .csv, has header, values and index. Value can be any
> types.

**Create a table:**
```matlab
LastName = {'Sanchez';'Johnson';'Li';'Diaz'};
Age = [38;43;38;40];
Weight = [176;163;131;133];
BloodPressure = [124 93; 109 77; 125 83; 117 75];
T = table(LastName,Age,Weight,BloodPressure)
```
     LastName      Age     Weight    BloodPressure
    ___________    ___     ______    _____________
    {'Sanchez'}    38       176       124     93  
    {'Johnson'}    43       163       109     77  
    {'Li'     }    38       131       125     83  
    {'Diaz'   }    40       133       117     75  
   
**Change table index's name (row's name):**
```matlab
T.Properties.RowNames = {'Name1','Name2','Name3','Name4'}
```
		     LastName      Age     Weight    BloodPressure
		    ___________    ___     ______    _____________
	Name1    {'Sanchez'}    38       176       124     93  
	Name2    {'Johnson'}    43       163       109     77  
	Name3    {'Li'     }    38       131       125     83  
	Name4    {'Diaz'   }    40       133       117     75 

**Change table column name:**
```matlab
T.Properties.VariableNames = {'Name1','Name2','Name3','Name4'}
```
**Add unit information:**
```matlab
T.Properties.VariableUnits = {'','year','kg',''}
```
**Add description for columns:**
```matlab
T.Properties.VariableDescription{'LastName'} = 'This is lastname'
T.Properties.VariableDescription = {'This is lastname' 'This is age' 'This is Weight' ''This is bloodpressure'}
```
**Accessing data in table:**
```matlab
T.LastName % return the rows of the collumn LastName
```
**Summary a table, will return variable names, description, units, min, max median:** `summary(T)`

**Sort rows:**
```matlab
sortrows(T,{'LastName','Age'},{'Ascend','Descend'})
```
**Read table from a file (.csv, .xlsx):** `T = readtable('file.csv')`

**Write table to a file:**
```matlab
writetable(T,'table.csv','Delimiter',',')
```
**Select, add, delete rows:**
```matlab
T(2,3) % select 1
T(1:2,:) % select all rows from 1-2
T(:,2:3) % select all cols from 2-3
T.Age % select all rows of column Age
T('IndexName1',:) % select all cols of IndexName1
T([true false true false],:) % select only rows specified as true
```
**Add row:**
```matlab
T2 = {{'John'},11,140,[123 88]}
T(end+1,:) = T2 % add 1 row at the end of the table
T = [T;T2] % add table T2 at the end of table T
```
**Remove row:**
```matlab
T('IndexName1',:) = []
```
**Add column:**
```matlab
T.newColName =  [1 2 3 4] % add new col at the end, must match the table height

T = [T;T2] 
% add columns T2 at the end
% col names must unique and T2's height must match T's height
```
**Remove column:**
```matlab
T.colName = []
T = T(:,1:end-1)
```
**Remove rows/cols with missing data:** `rmmissing(T,dim,Name,Value)`

**Sync 2 tables:** `synchronize(T,T2)`
- If 2 cols of 2 tables have the same col name and data, skip and store 1 only. 
- If 2 cols of 2 tables have the same col name and different data. add table name to that row (i.e Weight_T1, Weight_T2), store 2 cols.
- If 2 cols of 2 table are different, store 2 cols

### Struct

> A structure array is a data type that groups related data using data
> containers called fields. Each field can contain any type of data.

**Create a struct:** `struct(field,value)`
```matlab
student = struct('Name','','Grade',[],'Class','')
student(1).Name = 'John'
student(1).Grade = [10 9 10]
student(1).Class = 'A'
student(2).Name = 'Andy'
student(2).Grade = [10 9 10]
student(2).Class = 'A'
```
**Remove a field:** `rmfield(s,'fieldName')`

**Concat 2 structs:** 
```matlab
s = [s1;s2]
```
**Store fields in variables:**
```matlab
[v1 v2] = student.Name
v1 % -> 'John'
v2 % -> 'Andy'
A = {student.Name}
% -> { {'John'} {'Andy'} }
```
### Map

> A hash map that contain key-value pairs.

Create a map:
```matlab
mymap = containers.Map({'key1' 'key2' 'key3'},{'value1' 1 [1 2]})
```
Retrieve keys or values:
```matlab
mymap('key1') % -> 'value1'
keys(mymap) % -> return cell of all keys
values(mymap) % -> return cell of all values
```
Remove a key-value pair:
```matlab
remove(mymap,'key1')
```
Check if a map contains a key:
```matlab
isKey(mymap,{'key1' 'key4'}) % -> [1 0]
```
Concat 2 maps:
```matlab
mymap2 = containers.Map({'key4' 'key5'},{'value4' 3})
mymap = [mymap;mymap2]
```
[↑ Back to top](#table-of-contents)

## Programming
**If else statement:**
```matlab
a = 1
if a == 1
	disp("it's 1")
elseif a == 2
	disp("it's 2")
else 
	disp("it's not 1 or 2")
end
```
**For loop:**
```matlab
start = 1
step = 2
ends = 10
for i = start:step:ends
	disp(i)
end

for i = [1 3 5 7]
	disp(i)
end
```
**While loop:**
```matlab
i = 5
while i < 5
	i = i + 1
	disp(i)
end
```
**Break, continue loop:** `continue` `break`

**Switch case:**
```matlab
a = 2
switch a
	case 1
		disp("The number is 1")
	case 2
		disp("The number is 2")
	otherwise
		disp("Not 1 or 2")
end
	
```
### Function

**Define a function:**
```matlab
% Multiple outputs
function [outArg1,outArg2] = func_name(inArg1,inArg2)
	outArg1 = inArg1;
	outArg2 = inArg2;
end

[a b] = func_name(1,2)
```

[↑ Back to top](#table-of-contents)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg3NDIwNzM1LC0xMTg2NjE3NzMxLC0xNT
k5MTc5Mjc1LDE4ODIxNTc0ODAsLTI0NTMzOTkzNywtMTc2ODcy
MTk5MywxMTQyMjk2MjMsOTE1MDYxNzkyLC0xMDQ1Mzg3MzMzLC
0xMjc5NTQ4MzgsMTU5NzY3MTYyMiwxNzE1MjQ3Mzk4LDIwNDI1
NDc4NzMsMTc3Mjg5MDU5LDE3NzI4OTA1OSwtMTMyODg4MjcwNi
wtMjA1MTQwODY1MCwxMjc2OTIwMDcyLDQwNDUzNTA3MywtMTA1
NzIzNzcyN119
-->