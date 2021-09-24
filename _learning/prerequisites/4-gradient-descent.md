---
title: Gradient Descent 
slug: gradient-descent
background: '/img/bg-post.jpg'
lesson_number: 4
---

In the previous topic, we got ourselves a nice loss function to tell us how good or bad of a job we're doing at fitting a given line, of the form *f(x) = a.x + b*, to some data. We could write the expression of that function as follows:

<p align="center">
	<img src="/img/learning/prerequisites/sum-of-squares/sum_of_squares_5.png"/>
</p>

This is a particular case of the sum of squares, where the function we're using to try to fit the data is a line. Here, the input *x* is 1-dimensional, and the target variable is *y*. Using this expression for the sum of squares, we observe the difference between the real data *y* and the output of the line function, given by *a.x + b*. But now that we have something to measure how good our line is, how do we improve it to fit the data better? We will see that in just a bit, but for now...

### Function Weights

Remember when we said that *a* and *b* were the parameters of our line function? Well, they are actually more commonly referred to as the *weights* of the function. For that reason, instead of *a, b, ..., z*, we will henceforth refer to the weights of a function as *w<sub>0</sub>, w<sub>1</sub>, ..., w<sub>N</sub>*, when our data is *N*-dimensional.

Using our previous nomenclature, if the data had 1 dimensions, we would have 2 parameters: *a* (the slope) and *b* (the *y*-intercept). If the data had 2 dimensions, we would have the parameters corresponding to a plane equation: *a*, *b*, and *c*. This does, of course, generalize to any number of dimensions, which means that, if our data is *N*-dimensional, we have *N + 1* weights (or parameters).

Before, we described the expression (using the more compact vector notation) for the generic sum of squares error function (let's call it *E*), as follows:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_1.png"/>
</p>

For a *D*-dimensional input *x*, and using the new weight notation, our linear function *f* will look something like this:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_2.png"/>
</p>

Notice that sad little *w<sub>0</sub>* in there? Yeah, me too. The intercept term *w<sub>0</sub>* is called the *bias* parameter of our linear function. To make writing this expression easier, we "pretend" that there is an extra dimension *x<sub>0</sub> = 1* such that:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_3.png"/>
</p>

It should be clear by now why we call the linear function parameters *weights*: each of them scales a dimension of the data, making it contribute more or less to the summation. Of course, *w<sub>0</sub>* is always just multiplied by 1. Using the vector notation from before and our new weight nomenclature, the sum of squares error function looks like this:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_4.png"/>
</p>

It is common to use the *mean squared error*, or MSE, instead of just the summation of the errors, which will give you the same solution but [has a few advantages](https://stats.stackexchange.com/questions/158170/why-is-it-necessary-to-divide-by-the-number-of-samples-when-optimizing-squared-e). For reasons we'll see in a bit<sup>*\**</sup>, we also divide the MSE by 2. Finally, our squared error function looks like this:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_5.png"/>
</p>

### Gradient...

Now that we have our loss function properly defined, how do we minimize it?

You may remember from calculus that, when we want to find the minimum of a given function with respect to variable, we can calculate that function's derivative with respect to that variable, equate it to zero, and solve for that variable.

The weights (or parameters) are the variables we want to tweak in order to minimize our loss function, so we need to derive it with respect to **w**. Since **w** is a vector (*i.e.* may have multiple dimensions), we need to compute the *gradient* of *E* with respect to it:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_6.png"/>
</p>

You may remember from multivariate calculus that the gradient gives the direction of steepest ascent. In other words, each partial derivative in the above column vector will tell us, for that dimension of the weight vector, in what way the value of *E* will change if we increase it by a very small amount, while keeping the other dimensions of the weight vector fixed.

### ... Descent!

Gradient descent is an iterative algorithm. This means that we apply updates repeatedly until we achieve some desirable result. In particular, we will update the weight vector **w** over and over until we find the values for it which minimize function *E* the most. First, we need to compute the gradient of *E* with respect to **w**:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_7.png"/>
</p>

Now we have an expression that tells us how to change the values of **w** in the direction of steepest ascent. "But wait", you may ask, "weren't we trying to *minimize* the error function? We would we want to ascend, and not *de*scend?" - what a keen reader you are, and absolutely correct too! Here is how we write our gradient descent rule:

<p align="center">
	<img src="/img/learning/prerequisites/gradient-descent/gradient_descent_8.png"/>
</p>

The arrow means that the left side **w** is updated to the value on the right side. Since we want to *decrease* the error function as quickly as possible, we update the weights in the direction opposite the gradient. That is, by subtracting the gradient to the weights instead of adding it, we are following the *direction of steepest descent*. Again, we do this since we want to find the *minimum* of the error function *E*. Oh, that *alpha* there? It is a constant, known as the *learning rate*. The larger the learning rate is, the larger the step we will take in the direction of steepest descent. This value is not important for now, just know that it usually takes a small value such as 0.01 or 0.001.

The algorithm runs until an arbitrary number of consecutive iterations causes a small enough change in weights. When that happens, we have found our minimum, and have thus achieved *convergence*.

### Conclusions

We started calling the parameters of our linear functions *weights*. We went through gradient descent, an iterative algorithm that updates the weights of our linear function until it does the best possible job at fitting some data. We know it does a good job at fitting the data because we will have found the minimum of an error function which shows us how good or badly we're doing at that.

"What if I want to fit a nonlinear function to my data? Can I do that?" - yes! As we will see in the [*regression*](/learning/machine-learning/linear-regression) chapter, we can use gradient descent for all sorts of other functions. This extends to error functions too; we can use whichever one we want!

There are still two very important things to note:
- Gradient descent isn't guaranteed to find the global minimum of the error function
- We could actually compute the best possible weight values directly and deterministically

If those last two sentences left you scratching your head, don't worry: we will make sense of them in the next chapter, [*Optimization*](/learning/prerequisites/optimization).

<sup>*\**</sup> - When we differentiate the squared expression in the error function, we get a 2 in the derivative. To make the resulting derivative "cleaner", it is common to divide the squared error loss function by 2 for mathematical convenience.
