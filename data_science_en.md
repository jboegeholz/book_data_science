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


    import numpy as np


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


## Classification
### Quality
### Confusion Matrix
Let’s take the example

y_true = ["dog", "dog",     "non-dog", "non-dog", "dog", "dog"]
y_pred = ["dog", "non-dog", "dog",     "non-dog", "dog", "non-dog"]

When we look at the prediction we can count the correct and incorrect classifications:

    dog correctly classified as dog: 2 times (True Positive)
    non-dog incorrectly classified as dog: 1 time (False Positive)
    dog incorrectly classified as non-dog: 2 times (False Negative)
    non-dog correctly classified as non-dog: 1 time (True Negative)

When we visualize these results in a matrix we already have the confusion matrix:

![](images/confusion_matrix.png "")

#### sklearn

We can calculate the confusion matrix with sklearn in a very simple manner

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_true, y_pred, labels=["dog", "non-dog"]))

the output is:

    [[2 2]
    [1 1]]

which can be indeed confusing because the matrix is transposed. 
In contrast to our matrix from above the columns are the prediction and the rows are the actual values:

![](images/confusion_matrix_2.png "")

And that’s all – if you just have a binary classifier.
Multi-label classifier

So what happens, when your classifier can decide between three outcomes, say dog, cat and rabbit? (You can generate the test data with numpy random choice)

y_true = ['rabbit', 'dog', 'rabbit', 'cat', 'cat', 'cat', 'cat', 'dog', 'cat']
y_pred = ['rabbit', 'rabbit', 'dog', 'cat', 'dog', 'rabbit', 'dog', 'cat', 'dog']

cm = confusion_matrix(y_true, y_pred, labels=["dog", "rabbit", "cat"])

[[0 1 1]
[1 1 0]
[3 1 1]]

Precision and Recall

In the realms of Data Science you’ll encounter sooner or the later the terms “Precision” and “Recall”. But what do they mean?

Living together with little kids You very often run into classification issues:

My daughter really likes dogs, so seeing a dog is something positive. When she sees a normal dog e.g. a Labrador and proclaims: “Look, there is a dog!”

That’s a True Positive (TP)

If she now sees a fat cat and proclaims: “Look at the dog!” we call it a False Positive (FP), because her assumption of a positive outcome (a dog!) was false.

If I point at a small dog e.g. a Chihuahua and say “Look at the dog!” and she cries: “This is not a dog!” but indeed it is one, we call that a False negatives (FN)

And last but not least, if I show her a bird and we agree on the bird not being a dog we have a True Negative (TN)

This neat little matrix shows all of them in context:

![](images/precision_and_recall.png "")

If I show my daughter twenty pictures of cats and dogs (8 cat pictures and 12 dog pictures) and she identifies 10 as dogs but out of ten dogs there are actually 2 cats her precision is 8 / (8+2) = 4/5 or 80%.

    Precision = TP / (TP + FP)

![](images/precision.png "")

Knowing that there are actually 12 dog pictures and she misses 4 (false negatives) her recall is 8 / (8+4) = 2/3 or roughly 67%

    Recall = TP / (TP + FN)

![](images/recall.png "")

Which measure is more important?

It depends:

If you’re a dog lover it is better to have a high precision, when you are afraid of dogs say to avoid dogs, a higher recall is better 🙂


Different terms

Precision is also called Positive Predictive Value (PPV)

Recall often is also called

* True positive rate
* Sensitivity
* Probability of detection

Other interesting measures
Accuracy

    ACC = (TP + TN) / (TP + FP + TN + FN)

![](images/accuracy.png "")

F1-Score

You can combine Precision and Recall to a measure called F1-Score. It is the harmonic mean of precision and recall

    F1 = 2 / (1 / Precision + 1 / Recall)

#### Scikit-Learn

scikit-learn being a one-stop-shop for data scientists does of course offer functions for calculating precision and recall:

    from sklearn.metrics import precision_score

    y_true = ["dog", "dog", "not-a-dog", "not-a-dog", "dog", "dog"]
    y_pred = ["dog", "not-a-dog", "dog", "not-a-dog", "dog", "not-a-dog"]

    print(precision_score(y_true, y_predicted , pos_label="dog"))

Let’s assume we trained a binary classifier which can tell us “dog” or “not-a-dog”

In this example the precision is 0.666 or ~67% because in two third of the cases the algorithm was right when it predicted a dog

    from sklearn.metrics import recall_score

    print(recall_score(y_true, y_pred, pos_label="dog"))

The recall was just 0.5 or 50% because out of 4 dogs it just identified 2  correctly as dogs.

    from sklearn.metrics import accuracy_score

    print(accuracy_score(y_true, y_pred))

The accuracy was also just 50% because out of 6 items it made only 3 correct predictions.

    from sklearn.metrics import f1_score

    print(f1_score(y_true, y_pred, pos_label="dog"))

The F1 score is 0.57 – just between 0.5 and 0.666.

What other scores do you encounter? – stay tuned for the next episode 🙂

## scikit-learn