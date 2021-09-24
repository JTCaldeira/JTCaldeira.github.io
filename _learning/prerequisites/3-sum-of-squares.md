---
title: Sum of Squares
slug: sum-of-squares
background: '/img/bg-post.jpg'
lesson_number: 3
---


Say you worked really hard and got your hands on the following two-dimensional data:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_1.png"/>
</p>

We will use *x* and *y* to denote the first and second dimensions of the records, respectively.

Let us also refer to *y* as the *target variable*. From a statistics/machine learning point of view, a target variable is a variable (a dimension of the data) which we wish to be able to model and predict using other variables that describe the data. We use the word *sample*, or *record*, to refer to the set of variables which describe a certain object; and each sample in the dataset has a value for its target variable. In other words, when we use the word "sample", we will refer to all the dimensions of the data except for the one that we want to be able to model, that is, the target variable. In this case, our samples are *1*-dimensional, being described by the variable we called *x*. Usually, objects will be described by more features/variables, and thus samples will have more than just a single dimension (*x*). Furthermore, we usually refer to the set of samples and target values separately. In this case, for example, we could have a set of *x* values (samples) and another set of *y* values (target values for the samples). However, for simplicity in visualizing the data in a *2*-D plot, we're using *2*-tuples to represent the data points instead of separating it into two sets.

Let us draw a horizontal line that corresponds to the mean value of the *y* variable, whose equation is *y = 6.2*, and plot it alongside the data:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_2.png"/>
</p>

We usually like to plot lines that nicely show trends between variables, but the line we chose does a visibly bad job at that. However obvious that might look to us though, before we can try to choose a better one, we first need to know how good (or... bad) of a "fit" that line is to the data. We can do that by measuring the distance between the line and each data point.

We could choose the horizontal, perpendicular (*i.e.* orthogonal), or vertical distances between the line and the points. It is common to use regression to predict values of *y* (the target variable) from given values of *x*, so let us use the vertical distance. If we were trying to use the blue line to predict values of *y* from given values of *x*, this distance would represent the	*residual*, or *error*, in our prediction. The residuals between each point and the blue line are shown below as red dotted lines:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_3.png"/>
</p>

To measure how well the line fits the data, we see how close it is to the data points. In other words, we have to sum all the residuals to know how badly the line failed at fitting the data. The residual (or distance) between the first data point and the blue line is *r = 6.2 - 3 = 3.2*. So now we just do this to all the other data points, right? Well...

If we look at the 3rd data point and try to do the same thing, we will have *r = 6.2 - 8 = -1.8*. Since the result is negative, adding it with the other residuals will shrink down the total sum of residuals. That can't be!

Why not just take the absolute value of the difference between the blue line and each data point?

While that would solve the problem above, the absolute value is not continuously differentiable, which is a useful property for optimization techniques such as gradient descent (more on that later).

In practice, we square the residuals (or errors) since, among other reasons<sup>*\**</sup>, the squared difference is continuously differentiable; hence the name "sum of squared residuals", "sum of squared errors", or simply "sum of squares".

### Sum of Squares

Now that we have solved the problem above, we can calculate the sum of squared errors:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_4.png"/>
</p>

And that is our measure of how well the blue line fits the data. But can we come up with an expression that generalizes the sum of squares to other lines that aren't horizontal? Yes!

The generic line equation is *y = a.x + b*, where *a* is the slope of the line and *b* is its *y*-intercept. With *x<sub>n</sub>* being the *n*-th value of the set of *x* values, we can write a sum of squared errors for that simple line function:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_5.png"/>
</p>

Here, *a.x<sub>n</sub> + b* is the output of a line function for the *n*-th *x* value, and *y<sub>n</sub>* is the actual (real) value associated with that same input. Since the sum of squares quantifies the error we make when trying to fit a function (in this case, a line) to some data, we call it an "error function".

We have seen that in this error function, which is a particular case of the sum of squares where our "fitting" function is a line, we observe the difference between the real data *y* and the output of a line function, given by *a.x + b*. But now that we have something to measure how good our line is, how do we improve it to fit the data better? We will learn how to do that later, but for now...

The output of our line function is *f(x) = a.x + b*. We have no control over the values of *x*, as those are the samples of our dataset. We can, however, try to adjust *a* and *b* so that the loss function gives us a smaller value. For that reason, *a* and *b* are called *parameters* of our line function.

We can do this for other linear functions! If, for example, we had 3-dimensional data tuples *(x, y, z)* as opposed to just 2-dimensional data tuples *(x, y)*, we would be trying to fit a 3-D hyperplane to the data instead of a line. Making *z* our target variable, we write the formula for a 3-D plane in terms of *x* and *y* as follows: *f(x, y) = a.x + b.y + c*. We can write the sum of squares expression for such a plane:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_6.png"/>
</p>

However, if we're dealing with data that has a higher dimensionality, we're gonna have a bad time writing the sum of squares expression this way...

### Multivariate Sum of Squares

In the example above, we wrote the sum of squares formula in terms of functions that deal with 2-dimensional and 3-dimensional data (where one of those dimensions is the target), respectively. In real world scenarios, we may come across data that is described by several more variables. It would be useful, then, to generalize the sum of squares expression so that it applies to data of any dimensionality. The equation for a linear function which takes an *m*-dimensional input *x* can be written as *f(x) = a<sub>1</sub>.x<sub>1</sub> + a<sub>2</sub>.x<sub>2</sub> + ... + a<sub>m</sub>.x<sub>m</sub> + b*. Here, *x<sub>m</sub>* corresponds to the value of *x* for its *m*-th dimension, and *a<sub>m</sub>* is its coefficient. We will refer to an *m*-dimensional input, *(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>m</sub>)*, as **x** for simplicity. Now, we can use **x** to refer to an input vector with any dimensionality. We can then rewrite generalized the sum of squares as:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_7.png"/>
</p>

Now, **x**<sub>*n*</sub>, the *n*-th input value, is a vector with any amount of dimensions. We still use *y<sub>n</sub>* to refer to the real target value for the *n*-th sample.

### Mean Squared Error

A popular variation of the sum of squares is called the "mean squared error", or MSE for short.

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_8.png"/>
</p>

As the name suggests, the MSE over all *N* data points returns the *mean* (or average) error.

### Error Functions

In mathematical optimization, a loss function, error function, or cost function, is a function that maps an event to its "cost" (a real number). In optimization, the goal is usually to *minimize a loss function*. In contrast, for a few particular domains, it is more common to instead try to *maximize an objective function* (also called reward function, utility function, etc.). We will however, only deal with the former, since it is *far* more common in the fields herein discussed.

In statistics, a loss function is typically used to estimate parameters. It is a function of the difference between the values we predicted for some data and the true values for that data.

It may now start to make sense that the sum of squared errors is one such loss function. If we were trying to use a line to predict, given a value of *x* in our dataset, what its corresponding value of *y* would be, the sum of squares would tell us whether it does a good job or not by computing a loss based on the distance between the image of *x* in our line and its real *y* value.

### Conclusions

We looked at the "sum of squares" error function, also called "sum of squared errors", or just "squared errors". Here are a few final remarks regarding it.

I'd be remiss if I didn't mention the purpose of telling you about the sum of squares: it is the backbone of "least squares fitting", a standard approach in [*regression*](/learning/machine-learning/linear-regression). In least squares, we try to minimize the sum of squared errors between the data and whichever function we're trying to fit to it. We do that by tweaking the parameters of our function. In the case of the blue line we used as an example, we would adjust the slope *a* and *y*-intersept *b* to minimize the sum of squared errors. This concept extends to multivariable (or multidimension) contexts, where we may have a larger number of parameters to adjust.

<sup>*\**</sup> - Squaring the residuals/errors is useful because:
- Positive quadratic functions are *convex*, and a sum of convex functions is convex too. Convex functions are nice because they have a single global minimum, which means that "rolling downhill" from anywhere will take you to that global minimum. This means that optimization techniques, such as [*gradient descent*](/learning/prerequisites/gradient-descent), will always converge to the right answer.
- Least squares has a *closed-form solution*, which allows us to compute the right answer (*i.e.* the one that minimizes the sum of squared errors) *directly*. That is, we can compute the best result numerically rather than analytically. This would not be possible if we used the absolute value of the errors instead of squaring them.

If these last two paragraphs weren't all too clear to you, worry not. In the next pages in this section, we will discuss [optimization](/learning/prerequisites/optimization), wherein these concepts will be addressed and explained more thoroughly.
