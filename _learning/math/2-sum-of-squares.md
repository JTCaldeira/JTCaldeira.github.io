---
title: Sum of Squares
slug: sum-of-squares
background: '/img/bg-post.jpg'
lesson_number: 2
---

Say you worked really hard and got your hands on the following two-dimensional data:

<p align="center">
	<img src="/img/learning/math/sum_of_squares_1.png"/>
</p>

Let us use *x* and *y* to denote the first and second dimensions of the records, respectively.

Let us also consider a horizontal line that corresponds to the mean value of the *y* variable, whose equation is *y = 6.2*, and plot it alongside the data:

<p align="center">
	<img src="/img/learning/math/sum_of_squares_2.png"/>
</p>

We usually like to plot lines that nicely show trends between variables, but the line we chose does a visibly bad job at that. However obvious that might look to us though, before we can try to choose a better one, we first need to know how good (or... bad) of a "fit" that line is to the data. We can do that by measuring the distance between the line and each data point.

We could choose the horizontal, perpendicular (*i.e.* orthogonal), or vertical distances between the line and the points. It is common to use regression to predict values of *y* from given values of *x*, so let us use the vertical distance. If we were trying to use the blue line to predict values of *y* from given values of *x*, this distance would represent the	*residual*, or *error*, in our prediction. The residuals between each point and the blue line are shown below as red dotted lines:

<p align="center">
	<img src="/img/learning/math/sum_of_squares_3.png"/>
</p>

To measure how well the line fits the data, we see how close it is to the data points. In other words, we have to sum all the residuals to know how badly the line failed at fitting the data. The residual (or distance) between the first data point and the blue line is *r = 6.2 - 3 = 3.2*. So now we just do this to all the other data points, right? Well...

If we look at the 3rd data point and try to do the same thing, we will have *r = 6.2 - 8 = -1.8*. Since the result is negative, adding it with the other residuals will shrink down the total sum of residuals. That can't be!

Why not just take the absolute value of the difference between the blue line and each data point?

While that would solve the problem above, the absolute value is not continuously differentiable, which is a useful property for optimization techniques such as gradient descent (more on that later).

In practice, we square the residuals (or errors) since, among other reasons<sup>*\**</sup>, the squared difference is continuously differentiable; hence the name "sum of squared residuals", or "sum of squared errors", or simply "sum of squares".

### Sum of squares

Now that we have solved the problem above, we can calculate the sum of squared errors:

<p align="center">
	<img src="/img/learning/math/sum_of_squares_4.png"/>
</p>

And that is our measure of how well the blue line fits the data. But can we come up with an expression that generalizes the sum of squares to other lines that aren't horizontal? Yes!

The generic line equation is *y = a.x + b*, where *a* is the slope of the line and *b* is its *y*-intercept. With *x<sub>n</sub>* being the *n*-th value of the set of *x* values, we can write a sum of squared errors for that simple line function:

<p align="center">
	<img src="/img/learning/math/sum_of_squares_5.png"/>
</p>

Here, *a.x<sub>n</sub> + b* is the output of a line function for the *n*-th *x* value, and *y<sub>n</sub>* is the actual (real) value associated with that same input. Since the sum of squares quantifies the error we make when trying to fit a function (in this case, a line) to some data, we call it an "error function".

### Multivariate sum of squares

In the example above, we wrote the sum of squares formula in terms of a function that takes scalars as an input. However, in many real world scenarios, data will be described by several variables (or attributes), and as such will be represented as a multidimensional vector. In that case, considering a linear function that takes an *m*-dimensional vector *x*, its equation would look like *y = a<sub>1</sub>.x<sub>1</sub> + a<sub>2</sub>.x<sub>2</sub> + ... + a<sub>m</sub>.x<sub>m</sub> + b*, where *x<sub>m</sub>* corresponds to the value of *x* for its *m*-th dimension, and *a<sub>m</sub>* is its coefficient. We can, however, apply the sum of squares to nonlinear functions too, so let's use *f(***x***)* to refer to any type of function that we're using to try to fit our data. We can then rewrite the sum of squares as:

<p align="center">
	<img src="/img/learning/math/sum_of_squares_6.png"/>
</p>

Now, **x**<sub>*n*</sub>, the *n*-th input value to the function *f*, is no longer a scalar, but a vector with any number of dimensions.

### Mean squared error

A popular variation of the sum of squares is called the "mean squared error".

<p align="center">
	<img src="/img/learning/math/sum_of_squares_7.png"/>
</p>

As the name suggests, the mean squared error over all *N* data points returns the *mean* (or average) error.

### Conclusions

We looked at the "sum of squares" error function, also called "sum of squared errors", or just "squared errors". Here are a few final remarks regarding it.

I'd be remiss if I didn't mention the purpose of telling you about the sum of squares: it is the backbone of "least squares fitting", a standard approach in [*regression*](/learning/machine-learning/linear-regression). In least squares, we try to minimize the sum of squared errors between the data and whichever function we're trying to fit to it. We do that by tweaking the parameters of our function. In the case of the blue line we used as an example, we would adjust the slope *a* and *y*-intersept *b* to minimize the sum of squared errors. This concept extends to multivariable (or multidimension) contexts, where we may have a larger number of parameters to adjust.

<sup>*\**</sup> - Squaring the residuals/errors is useful because:
- Positive quadratic functions are *convex*, and a sum of convex functions is convex too. Convex functions are nice because they have a single global minimum, which means that "rolling downhill" from anywhere will take you to that global minimum. This means that optimization techniques, such as *gradient descent*, will always converge to the right answer.
- Least squares has a *closed-form solution*, which allows us to compute the right answer (*i.e.* the one that minimizes the sum of squared errors) *directly*. That is, we can compute the best result numerically rather than analytically. This would not be possible if we used the absolute value of the errors instead of squaring them.

If these last two paragraphs weren't all too clear to you, worry not. I will make an article about [optimization](/learning/math/optimization), wherein these concepts will be addressed and explained more thoroughly.