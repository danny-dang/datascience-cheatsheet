 ## Installation
 ```python
 pip install matplotlib
 ```
```python
import matplotlib.pyplot as plt
 ```
[Tutorial](https://matplotlib.org/tutorials/index.html)

 ## Ploting
[matplotlib.pyplot.plot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot)

`plt.plot(*args, scalex=True, scaley=True, data=None, **kwargs)`

Ploting x, y, Labeling, Title, Legend:
```python
ages_x = [18, 19, 20, 21, 22, 23]

py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732]
plt.plot(ages_x, py_dev_y, label='Python')

js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000]
plt.plot(ages_x, js_dev_y, label='JavaScript')

dev_y = [17784, 16500, 18012, 20628, 25206, 30252]
plt.plot(ages_x, dev_y, color='#444444', linestyle='--', label='All Devs')

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')

plt.legend()
plt.grid()

plt.show()
```
![enter image description here](https://i.imgur.com/napfBE2.png)

## Bar chart
[matplotlib.pyplot.bar](https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py)

`plt.bar(x, height, width=0.8, bottom=None, align='center', data=None, **kwargs)`
- x, location of the bar in the x-axis
- height: height of the bar
- width: width of the bar
```python
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men = [20, 34, 30, 35, 27]
women= [25, 32, 34, 20, 25]

x_index = np.arange(len(labels)) #create equal x-axis indexes 
width = 0.35 

plt.bar(x_index - width/2, men, width = width, label= 'Men')
plt.bar(x_index + width/2, women, width = width, label= 'Women')

plt.xlabel('Groups')
plt.ylabel('Count')
plt.title('Number of men and women in group')

plt.legend()
plt.xticks(ticks = x_index, labels = labels) # label the indexes
plt.show()
```
![enter image description here](https://i.imgur.com/NdMoPPZ.png)

The same with bar horizontal:
```python

labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men = [20, 34, 30, 35, 27]
women= [25, 32, 34, 20, 25]

y_index = np.arange(len(labels))
height = 0.35

plt.barh(y_index - height/2, men, height = height, label= 'Men')
plt.barh(y_index + height/2, women,height = height, label= 'Women')

plt.xlabel('Groups')
plt.ylabel('Count')
plt.title('Number of men and women in group')

plt.legend()
plt.yticks(ticks = y_index, labels = labels)

plt.show()
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0Nzk4MzU1MjddfQ==
-->