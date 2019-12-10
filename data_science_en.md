## Foreword

## The difference between Data Science, Machine Learning, Deep Learning and AI
A picture says more than a thousand words.

![DataScience-Overview](images/data_science_vs_ml.png "")

Data Science tries to answer one of the following questions:

* Classification -> “Is it A or B?”
* Clustering -> “Are there groups which belong together?”
* Regression -> “How will it develop in the future?”
* Association -> “What is happening very often together?”
## Statistics
### Bayes Theorem

## Numpy
Numpy is a package for scientific computing in Python.

```python
import numpy as np
```

The most important data structure is ndarray, which is short for n-dimensional array.

You can convert a list to an numpy array with the array-method
```
my_list = [1, 2, 3, 4]
my_array = np.array(my_list)
```
You can also convert an array back to a list with
```
my_new_list = my_array.tolist()
```
You can retrieve the dimensionality of an array with the ndim property:
```
my_array.ndim
```
and get the number of data points with the shape property
```
my_array.shape
```

### Vector arithmetic
#### Addition / Subtraction
```
a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])
a + b
array([5, 5, 5, 5])
a - b
array([-3, -1,  1,  3])
```

#### Scalar Multiplication

```
a = np.array([1, 2, 3, 4])
a * 3
array([3,  6,  9, 12])
```

To see why it is charming to use numpy’s array for this operation You have to consider the alternative:

```
c = [1,2,3,4]
d = [x * 3 for x in c]
```

Dot Product

```
a = np.array([1,2,3,4]) 
b = np.array([4,3,2,1])

a.dot(b)
```
20 # 1*3 + 2*3 + 3*2 + 4*1

### linspace function

To create e.g. x-axis indices you can use the linspace function from numpy.
You give it a range (e.g. 0 to 10) and the number of divisions and it will distribute the values evenly across that range. The stop values is included in the resulting value array by default.

Example:
```python
import numpy as np
np.linspace(0, 10, num=9)

array([ 0. , 1.25, 2.5 , 3.75, 5. , 6.25, 7.5 , 8.75, 10. ]
```

## Matplotlib
matplotlib is the workhorse of data science visualization. The module pyplot gives us MATLAB like plots.

The most basic plot is done with the “plot”-function. It looks like this:

![](images/pyplot_plot_01.png "")


```python
import matplotlib.pyplot as plt
plt.plot([0, 1, 2, 3], [0, 1, 2, 3])
plt.show()
```
The plot function takes an x and y array and draws a blue line through all points.

You can of course draw each point independently without a line:

```python
plt.plot([0, 1, 2, 3], [0, 1, 2, 3], "o")
```
![](images/pyplot_plot_03.png "")

or you can highlight the individual points while drawing the line

```python
plt.plot([0, 1, 2, 3], [0, 1, 2, 3], marker='o')
```
### Color of the plot

The color of the line can be changed with the color parameter. The default color of a plot is blue. If you want e.g. red as the color, you can use ‘r’.

```python
plt.plot([0, 1, 2, 3], [0, 1, 2, 3], color="r")
```

![](images/pyplot_plot_04.png "")

The following table shows the basic colors

| Color         | Shortcut      | 
| ------------- | ------------- | 
| blue          | b             |
| green         | g             |
| cyan          | c            |
| magenta       | m             |
| yellow        | y             |
| black         | k (not so obviuous)             |
| white         | w             |


In /matplotlib/_color_data.py you find additional colors, even colors from the XKCD color survey results

plt.plot(x, y, color="xkcd:nasty green")

```
![](images/pyplot_plot_05.png "")
```

### Stroke width and style

changing the width of the plotted line is done via linewidth

```
plt.plot([0, 1, 2, 3], [0, 1, 2, 3], linewidth=7.0, color="xkcd:nasty green")
```

![](images/pyplot_plot_06.png "")

and the stroke style can be altered with the linestyle parameter

```
plt.plot([0, 1, 2, 3], [0, 1, 2, 3], linestyle=":", color="xkcd:nasty green", linewidth=7.0)
```

![](images/pyplot_plot_07.png "")

### Axis Labels

In school I learned that all axis of a plot must have labels. So let’s add them:

plt.ylabel('some other numbers')
plt.xlabel('some numbers')

![](images/pyplot_plot_08.png "")

### Saving the plot

If You want to save the plot as a png you can replace the show command with

plt.savefig('scatter_01.png')

### Scatterplot

Here is another type of plot used in data science.

A very basic visualization is the scatter plot:

import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.random.rand(N)
y = np.random.rand(N)

plt.scatter(x, y)
plt.show()

![](images/scatter_01.png "")

### Multiple Plots

You might have wondered how to draw more than one line or curve into on plot. I will show you now.

To make it a bit more interesting we generate two functions: sine and cosine. We generate our x-values with numpy’s linspace function

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi)

sin = np.sin(x)
cos = np.cos(x)

plt.plot(x, sin, color='b')
plt.plot(x, cos, color='r')
plt.show()

You can plot two or more curves by repeatedly calling the plot method.

![](images/pyplot_plot_21.png "")

That’s fine as long as the individual plots share the same axis-description and values.

### Subplots

fig = plt.figure()
p1 = fig.add_subplot(2, 1, 1)
p2 = fig.add_subplot(2, 1, 2)
p1.plot(x, sin, c='b')
p2.plot(x, cos, c='r'

The add_subplot method allows us to put many plots into one “parent” plot aka figure. 
The arguments are (number_of_rows, number_of_columns, place in the matrix) 
So in this example we have 2 rows in 1 column, sine is in first, cosine in second position:

![](images/pyplot_plot_22.png "")

when you have a 2 by 2 matrix it is counted from columns to row

fig = plt.figure()
p1 = fig.add_subplot(221)
p2 = fig.add_subplot(222)
p3 = fig.add_subplot(223)
p4 = fig.add_subplot(224)
p1.plot(x, sin, c='b')
p2.plot(x, cos, c='r')
p3.plot(x, -sin, c='g')
p4.plot(x, -cos, c='y')

![](images/pyplot_plot_23.png "")

## Pandas


## Machine Learning

The Essence of Machine Learning

* A pattern exists
* The pattern cannot be described mathematically
* We have (enough) data on this problem


## The difference between supervised and unsupervised learning
Supervised Learning

You have training and test data with labels. 
Labels tell You to which e.g. class a certain data item belongs. 
Imagine you have images of pets and the labels are the name of the pets.

Unsupervised Learning

Your data doesn’t have labels. Your algorithm e.g. k-means clustering need to figure out a structure given only the data

## scikit-learn