---
title: Linear Regression
slug: linear-regression
background: '/img/bg-post.jpg'
lesson_number: 3
---

Say you worked really hard and got your hands on the following data:

$$
D = \{(1, 3), (2, 5), (3, 8), (4, 6), (5, 9)\}
$$

In this case, we're depicting the data as a set of 2-tuples where the second element is the target variable. Don't know what a target variable is? Give the [intro](/_learning/machine-learning/1-intro-to-ml.md) a read.

Since the second element of each tuple is that record's target value, that means our samples are 1-dimensional. We can also represent the data as we have done before, where the records are rows in a matrix and the target variables are stored in a column vector, where each element corresponds to a row in the record matrix:

$$
\v{X} = \begin{bmatrix}1\\2\\3\\4\\5\end{bmatrix},\quad \v{y} = \begin{bmatrix}3\\5\\8\\6\\9\end{bmatrix}
$$

Now; we usually like to plot lines that nicely show trends between variables. For example, we might use that to predict the value of some variable (target variable) given values for other recorded variables. In this case, since we are trying to predict a continuous numerical variable, this task is known as regression - which again, you should know if you've read the [intro](/_learning/machine-learning/1-intro-to-ml.md) (no hard feelings if you haven't). We might even want to predict the target variable of *our* dataset (wink wink). In fact, since we are using a linear function (a line, duh) to fit the data, we are trying to do *linear regression*. Let us draw a horizontal line that corresponds to the mean of the target variable, whose equation is $f(x) = 6.2$, and plot it alongside the data points:

<p align="center">
	<img src="/img/learning/machine-learning/linear-regression/line-horizontal.png"/>
</p>

Well... This line does a visibly bad job at showing a trend between these two variables, but however obvious that might be to us, before we can try to choose a better one, we first need to know how good (or bad) of a "fit" that line is to the data; this way, we might be able to tell the line how to improve. We can do that by measuring the distance between the line and each data point - that'll tell us the difference between each *real* value of the target variable and what the line represents.

We could choose the horizontal, perpendicular (i.e., orthogonal), or vertical distances between the line and the points. It is common to use regression to predict values of $y$ (the target variable) from given values of data points $x$, so let us use the vertical distance. If we were trying to use the blue line to predict values of $y$ from given values of $x$, this distance would represent the *residual*, or *error*, in our prediction. The residuals between each point and the blue line are shown below as red dotted lines:

<p align="center">
	<img src="/img/learning/machine-learning/linear-regression/line-residuals.png"/>
</p>

To measure how well the line fits the data, we see how close it is to the data points. In other words, we have to sum all the residuals to know how badly the line failed at fitting the data. The residual (or distance) between the first data point and the blue line is $r = 6.2 - 3 = 3.2$. So now we just do this to all the other data points, right? Well...

If we look at the 3rd data point and try to do the same thing, we will have $r = 6.2 - 8 = -1.8$. Since the result is negative, adding it with the other residuals will shrink down the total sum of residuals. That can't be!

Why not just take the absolute value of the difference between the blue line and each data point?

While that would solve the problem above, the absolute value is not continuously differentiable. Furthermore, quadratic functions are *convex*, and a sum of convex functions is convex too. These are both useful properties for optimization techniques such as gradient descent (more on that later). Thus, in practice, we square the residuals (or errors). So now we calculate the sum of the squared errors that our line makes when trying to represent the data points:

$$
Error = (6.2 - 3.0)^2 + (6.2 - 5.0)^2 + (6.2 - 8.0)^2 + (6.2 - 6.0)^2 + (6.2 - 9.0)^2 = 22.8
$$

\- but wait, we can generalize this computation for any number of data points. Let's say that the line corresponds to a function $f$, and that our dataset $D$ consists of $M$ tuples $(x, y)$, where $y$ is the target variable. Then, we could write a more general formula for calculating the sum of squared errors:

\begin{equation}\label{ss}
Error(D, f) = \sum_{i = 1}^{M} (f(x_i) - y_i)^2
\end{equation}

We have now achieved what we set out to do: we have a function - an *error function* - which quantifies how well the line drawn by $f$ fits our dataset. This error function is known as the *sum of squares*, or *sum of squared errors*, for obvious reasons. Let's now put our dataset aside for a second and talk a bit about error functions.

## Error Functions

In mathematical optimization, a *loss function*, *error function*, or *cost function*, is a function that maps an event to its "cost" (a real number). In optimization, the goal is usually to *minimize a loss function*. In contrast, for a few particular domains, it is more common to instead try to *maximize an objective function* (also called *reward function*, *utility function*, etc.). We will however, only deal with the former, since it is far more common in the fields herein discussed.

In statistics, a loss function is typically used to estimate parameters. It is a function of the difference between the values we predicted for some data and the true values for that data.

It may now start to make sense that the sum of squared errors is one such loss function - or *error* function, as we called it. If we were trying to use a line to predict, given a value of $x$ in our dataset, what its corresponding value of $y$ would be, the sum of squares would tell us whether it does a good job or not by computing a loss based on the distance between the image of $x$ in our line and its real $y$ value.

## Sum of Squares

Let's now return to our data, the line that we are trying to use to fit to the data, and the function we chose to measure how well it's doing its job (Equation \ref{ss}).

First of all, recall that the generic line equation is $f(x) = ax + b$, where $a$ is the slope of the line and $b$ its *y-intercept*. We can then write the sum of squared errors for this simple line function, considering the same dataset:

\begin{equation}\label{ss-line}
Error(D, f) = \sum_{i = 1}^{M} ((ax_i + b) - y_i)^2
\end{equation}

This is really the same thing that we did before, except now we replaced $f(x)$ by the actual "body" of the function. But what if we had higher-dimensional data? For example, what if our data were tuples $(x_1, x_2, y)$, where $y$ is still our target variable? Then, you might imagine that we would be trying to fit a 3D plane to the data, instead of a line. In that case, our sum of squares function would look like this:

\begin{equation}\label{ss-plane}
Error(D, f) = \sum_{i = 1}^{M} ((ax_{i,1} + bx_{i,2} + c) - y_i)^2
\end{equation}

However, if we're dealing with data that has even higher dimensionality, we're gonna have a bad time writing the sum of squares expression this way...

## Multivariate Sum of Squares

In the example above, we wrote the sum of squares formula in terms of functions that deal with 2-dimensional (Equation \ref{ss-line}) and 3-dimensional (Equation \ref{ss-plane}) data, where one of those dimensions is the target. In real world scenarios, we may come across data that is described by several more variables. It would be useful, then, to generalize the sum of squares expression so that it applies to data of any dimensionality.

First, let us recall the line equation $f(x) = ax + b$. Here, $a, b$ are the *parameters* of the the function $f$ since they determine the output of the function given some input. In other words, $a, b$ *parametrize* $f$. In the machine learning literature literature, we often see function parameters represented by the greek letter $\theta$, such that we would have $f(x) = \theta_1x + \theta_0$. If $f$ was the equation of a 3D plane, its function would be $f(x_1, x_2) = \theta_0 + \theta_1x_1 + \theta_2x_2$. I think you know where we're going with this. To describe a linear function whose input is $D$-dimensional, we have the general function

\begin{equation}
f(\v{x}) = \sum_{i = 1}^{D} \theta_0 + \theta_ix_i
\end{equation}

Notice that we use $\v{x}$ to represent the input to the function. Indeed, we usually denote input that has several dimensions as a vector, and $\v{x}$ is the a $D$-dimensional column vector whose elements are the values of that input.

We can go a step further: why not also "vectorize" the parameters of the function? That's indeed what we usually do! There is a caveat, though. Notice how there is always one more parameter than there are input dimensions. That extra parameter $\theta_0$ is to the intercept term (also called *bias*), and to make up for it, we "pretend" that there is an extra dimension in the input $x_0 = 1$. So, finally, we can write a compact general expression for linear functions:

\begin{equation}
f(\v{x}) = \v{\theta}^\top \v{x}
\end{equation}

Notice that both $\theta, \v{x}$ are column vectors, i.e., they have dimension $(D + 1, 1)$. In order to scale each element of the input, we need to multiply it by some parameter. This is why we take the transpose of the parameter vector: so that the multiplication is valid, and so that result is a scalar. If we instead took the transpose of $\vec{x}$, the result would be a $(D + 1, D + 1)$ matrix. If you're confused about this last part, you might need to brush up on your linear algebra.

So if we now write the sum of squared errors (SSE) where we have a multivariate linear function instead of just a line or a plane, we have:

\begin{equation}
SSE(D, \v{\theta}) = \sum_{i = 1}^{M} (\v{\theta}^\top \v{x}_i - y_i)^2
\end{equation}

Notice how we don't say $f$ is an argument to the error function, but rather its parameters $\v{\theta}$. This is somewhat of an arbitrary choice, really, but it'll make sense later. Notice also how we've been writing expressions for *linear* functions. We can actually apply the sum of squares to *any* type of function that we're using to try to fit some data. Let $f_\v{\theta}$ be one such function, parameterized by $\v{\theta}$. We write the general expression for the multivariate sum of squares:

\begin{equation}
SSE(D, f_\v{\theta}) = \sum_{i = 1}^{M} (f_\v{\theta}(\v{x}_i) - y_i)^2
\end{equation}

A popular variation of the sum of squares is called the *mean squared error*, or MSE for short.

\begin{equation}
MSE(D, f_\v{\theta}) = \frac{1}{M} \sum_{i = 1}^{M} (f_\v{\theta}(\v{x}_i) - y_i)^2
\end{equation}

As the name suggests, the MSE over all $M$ data points returns the *mean* (or average) error, so it is really just the sum of squares divided by the number of samples $M$.

Great; so we've done what we set out to do, which was to find a function - an error function - that told us how bad of a job our line was doing at fitting the data. In fact, we went super fancy and came up with the *sum of squared errors*, which works for *any* function that we are trying to use to fit our data. In fact, when we are trying to choose a function that learns to represent or learn something about data, we call it a *model*. In this case, our model is a line. However, the whole point of finding an error function was to actually use it to guide our model to improve - and we can do just that.

Indeed, if we could change the parameters of our line function so as to minimize the sum of squared errors, we would reach the best fit for the data. The method by which we minimize the sum of squared errors in a regression task (such as this one) is called the *least squares* method.

## Solving the Least Squares Problem

Remember how we said that squaring the errors instead of taking their absolute value was useful? One of the reasons for that is that the sum of squares is convex, which means that its minimum corresponds to a point where its derivative equals 0. The other reason for the squaring being useful was that a quadratic function is continuously differentiable - now ain't that convenient?

Recall that we want to find the parameters $\v{\theta}$ of $f$ for which the sum of squared errors (SSE) is minimal. In other words, we want to know the parameters $\v{\theta}$ for which the derivative of the SSE with respect to those parameters is 0. Therefore, we solve

\begin{equation}
\frac{\partial SSE(D, f_\v{\theta})}{\partial \v{\theta}} = 0
\end{equation}

Recall that we can use a matrix $\v{X}$, where the each row is a record and the columns are the features, to represent the set of samples of the dataset. Similarly, we can use $\v{y}$ to represent the target values of the respective records in $\v{X}$. An example of this representation was given when we first introduced the dataset (top of the page). Let us first look at the expression for the sum of squared errors using that notation:

\begin{equation}
SSE(D, f_\v{\theta}) = \Vert\v{X}\v{\theta} - \v{y}\Vert^2 = (\v{X}\v{\theta} - \v{y})^\top(\v{X}\v{\theta} - \v{y}) = \v{\theta}^\top\v{X}^\top\v{X}\v{\theta} - \v{\theta}^\top\v{X}^\top\v{y} - \v{y}^\top\v{X}\v{\theta} + \v{y}^\top\v{y}
\end{equation}

Notice how $\v{X}\v{\theta} - \v{y}$ results in an $(m \times 1)$ vector. While we would usually use a sum over all the samples, since we're now dealing with matrices and vectors, we get the $l-2$ (euclidean) norm over that $(m \times 1)$ resulting vector, which is equivalent to squaring each error and summing them all up. Now we differentiate the sum of squares with respect to the parameters:

\begin{equation}\label{ls-sol}
\frac{\partial SSE(D, f_\v{\theta})}{\partial \v{\theta}} = \frac{\partial (\v{\theta}^\top\v{X}^\top\v{X}\v{\theta} - \v{\theta}^\top\v{X}^\top\v{y} - \v{y}^\top\v{X}\v{\theta} + \v{y}^\top\v{y})}{\partial \v{\theta}} = 2\v{X}^\top\v{X}\v{\theta} - 2\v{X}^\top\v{y}
\end{equation}

Now that we have the derivative of our the error function with respect to the parameters $\v{\theta}$ of our model, we can find its minimum:

\begin{equation}
\frac{\partial SSE(D, f_\v{\theta})}{\partial \v{\theta}} = 0 \implies 2\v{X}^\top\v{X}\v{\theta} - 2\v{X}^\top\v{y} = 0 \implies \v{X}^\top\v{X}\v{\theta} = \v{X}^\top\v{y} \implies \v{\theta} = (\v{X}^\top\v{X})^{-1}\v{X}^\top\v{y}
\end{equation}

Note that we went through all this under the assumption that our model $f$ was of the form $\v{X}\v{\theta}$, i.e., its parameters scale all the elements of the input *linearly*. For that reason, we have actually been looking at *linear least squares*, or *ordinary least squares* (OLS). All OLS problems have a *closed-form solution*, which means that we were able to compute the right answer (i.e., the best $\v{\theta}$) directly. When our model is a nonlinear function, such as a neural network, we have a *nonlinear least squares* problem, and there isn't a closed-form solution for most such cases.

Using the shiny new formula we got in Equation \ref{ls-sol}, we can find the best parameters for our line function $f$ by plugging in the values for $\v{X}, \v{y}$ of our dataset:

$$
\theta = \begin{bmatrix}
    2.3\\
    1.3
\end{bmatrix},\quad\text{and so } f_\theta(x) = 1.3x + 2.3
$$

Let us now plot our line function alongside the data:

<p align="center">
	<img src="/img/learning/machine-learning/linear-regression/line-solved.png"/>
</p>

*Beautiful, isn't it?* - that's it for linear regression! Or is it...?

## Conclusion

Ordinary (or linear) least squares (OLS) is a method for (linear) regression, and we've seen how to derive the optimal parameters of a linear model in order to fit some data (Equation \ref{ls-sol}). We were able to directly calculate the parameters $\theta$ using a closed-form solution for OLS, which was pretty neat.

But what if I told you that it is actually rather uncommon to use this solution, and instead use a whacky algorithm that *doesn't* compute the solution directly? Check out the upcoming topic on *gradient descent* to find out more.

There are other forms of regression out there. We can really use *any* function we'd like to fit data: exponential functions, polynomial functions, sinusoidal functions, etc., and these forms of nonlinear regression deserve a topic of their own.

Finally, we also introduced some intuition for *error functions*, of which the sum of squared errors is an example. Error/loss/cost functions are central in the fields of artificial intelligence, optimization, operations research, and a whole lot more stuff.
