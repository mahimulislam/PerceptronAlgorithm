# PerceptronAlgorithm
Perceptron algorithm for finding the weights of  a Linear Discriminant function.
Problem Description:
1. Take input from “train.txt” file. Plot all sample points from both classes, but samples
from the same class should have the same color and marker. Observe if these two
classes can be separated with a linear boundary.
2. Consider the case of a second order polynomial discriminant function. Generate the
high dimensional sample points y.
3. Use Perceptron Algorithm (both one at a time and many at a time) for finding the weight-
coefficients of the discriminant function (i.e., values of w) boundary for your linear
classifier in task 2.
Here α is the learning rate and 0 < α ≤ 1.
4. Three initial weights have to be used (all one, all zero, randomly initialized with seed
fixed). For all of these three cases vary the learning rate between 0.1 and 1 with step size
0.1. Create a table which should contain your learning rate, number of iterations for one
at a time and batch Perceptron for all of the three initial weights. You also have to create
a bar chart visualizing your table data.
