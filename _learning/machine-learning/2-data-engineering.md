---
title: Data Engineering
slug: data-engineering
background: '/img/bg-post.jpg'
lesson_number: 2
---

> My data science teacher would kill me if they knew that I was devoting a single topic to this.

Addressing a problem with data science usually follows these steps:
- Asking some question; or, in statistics lingo, formulating a hypothesis.
- Getting data. In doing so, we must be aware of the following concerns:
    + Sampling. Do the recorded data truly represent the target population in question?
    + Relevancy. Are we measuring what we think are relevant variables, and are we including bogus information?
    + Privacy. Are we complying to relevant data protection and regulation standards?
- Engineering data. In this step, we begin by visualizing and profiling the data. We search for anomalies such as outliers, missing values, and other useful information. We then preprocess the data so as to address the anomalies found, and possibly improve it by performing feature engineering.
- Learning from data. Here is where machine learning mostly comes in: data scientists use our fancy models on their shiny data to do cool things. We also need to validate the models that we build - we will discuss that later.
- Evaluating the results. What metrics can we use to assess our results? Did we answer the question, or at least helped to do so?

Some also call this the *KDD process* - where KDD stands for "knowledge discovery in databases" - though this is older nomenclature.

## Data Profiling

What questions should we be asking about our data, and what should we be looking for, exactly? We address those questions by profiling our data, or by doing *data exploration*. In particular, we look at:
- The *dimensionality* of the data.
- The *distribution* of the variables that describe the data.
- The level of detail, or *granularity*, of each variable.
- Anomalies in the data.

##### Dimensionality

The dimensionality of the data is, quite literally, the number of dimensions (i.e., attributes) that describe them. In that case, we want the data to be as high-dimensional as possible, because that means we have more information about each sample, right? Well...

As the number of dimensions increases, so does the volume of the space in question. In fact, if we do not have enough samples to keep up with the rapid expansion in space volume, the data become highly sparse in that space. This is known as the *curse of dimensionality*, and we can visualize it by taking a dataset of 8 points in 1, 2, and 3 dimensions:

<p align="center">
	<img src="/img/learning/machine-learning/data-eng/curse-of-dimensionality.png"/>
</p>

The "Hughes phenomenon" - which is used (incorrectly, in my opinion) interchangeably with the curse of dimensionality - states that, for the same amount of samples, increasing the number of dimensions is improves the model's predictive power until it reaches some maximum value. From there, the power of the model *decreases*.

The takeaway from this section should be that you should always take into account the dimensionality of the data *versus* the amount of records. As we will see later, there are some methods for changing the number of features of the data in order to address some issues with respect to dimensionality.

##### Distribution

When discussing data distribution, some of the first statistics that we look for are those of *centrality*, i.e., that represent the middle of the data. Centrality measures include the mean, median, and mode.

Another important statistic is *population standard deviation*, which measures the dispersion of the data with respect to its mean. It is usually represented by the greek letter sigma $\sigma$, and is not to be confused with the *sample standard deviation*, which is usually represented by the letter *s*. Finally, the *variance* of a variable is the square of its standard deviation. The formulas for calculating these three statistics (assuming $N$ records, a population mean $\mu$ and sample mean $\bar{x}$) are:

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i = 1}^{N} (x_i - \mu)^2},\quad s = \sqrt{\frac{1}{N - 1} \sum_{i = 1}^{N} (x_i - \bar{x})^2},\quad Var = \sigma^2
$$

> Note: you might be wondering why the $N$ and $N - 1$ on the population and sample standard deviations, respectively. The StatQuest channel on YouTube has a [video](https://www.youtube.com/watch?v=sHRBg6BhKjI) doing a better job than I ever could as to why we do this.

This [page](https://www.mathsisfun.com/data/standard-deviation.html) has a cool explanation of the above three measures of dispersion. The standard deviation has the advantage of being expressed in the same units as the variable itself, which makes it more easily interpretable. But why do we need variance? Well, the squared standard deviation is often nicer to work with mathematically - you're gonna have to take my word for this one.

Finally, another thing you might notice is that when calculating the standard deviation, we don't take the absolute value of the difference between each sample and the mean. That would be called the *absolute mean deviation*. ([Here](https://emilkirkegaard.dk/en/wp-content/uploads/Revisiting-a-90-year-old-debate-the-advantages-of-the-mean-deviation.pdf) is an interesting read as to why it could be superior to standard deviation in some aspects.) Instead, we take the squared differences and compute the square root at the end. This emphasizes greater differences, i.e., outliers have greater impact on the measure, which can be helpful or not depending on the context.

Another common way of getting a sense of the distribution of a variable is to look at its *five-number summary*. Those five numbers are:
- The minimum value.
- The first quartile - Q1 - or $25^{th}$ percentile.
- The second quartile - Q2 - or $50^{th}$ percentile. This is the median of the variable.
- The third quartile - Q3 - or $75^{th}$ percentile.
- The maximum value.

Recall that quartiles divide (sorted) data into (in this case, four) quantiles of equal size. In other words, there are 25% of data points between each quartile. Similarly, 25% of the data points are between the minimum and Q1; *idem* for Q3 and the maximum.

The five-number summary is also the basis for drawing a [box plot](https://en.wikipedia.org/wiki/Box_plot).

While we're on the topic of data visualization, how about the [histogram](https://en.wikipedia.org/wiki/Histogram)? To some level of accuracy, a histogram plots the relative frequency of values of some variable. Indeed, we usually pick intervals of equal size (called *bins*) between the minimum and maximum values and, for each of those intervals, we count how many values for that variable "fall into" it. Furthermore, we can affect the detail with which we see value frequency: for example, by choosing smaller bin sizes, we see with "greater resolution" over value frequencies.

##### Granularity

The granularity with respect to a given variable is simply the level of detail that that variable describes, and the *finer* granularity is, the more detail it provides. Conversely, the *coarser* the granularity of some variable is, the less specific it is. For example, a variable that stores information about time in days has finer granularity than if it stored it in months or years. Obviously, you may aggregate data from some fine granularity onto a coarser level.

We can decrease the granularity level of a numerical variable by *discretizing* it. In general, discretization is the transformation of a continuous variable into a discrete one. In practice, we take a continuous variable and group it into intervals or categories; we usually do it in one of these three ways:
- Group by quantile: all the intervals have the same number of data points.
- Group by values: akin to what we usually do in histograms, we choose a fixed span of values (cf. a "bin").
- Cluster: use some clustering algorithm, e.g., *k-means*, to group the samples into some number of clusters.

We may decrease the granularity level of categorical variables by looking at some hierarchy in their values. Some examples are variables that describe time (days, months, years), location (street, city, country), and biological taxonomy (species, genus, family).

##### Anomalies

One type of data anomaly that needs addressing in order to work with most statistical and machine learning methods is that of missing values. For example, in the same record, values for some variables may take some NaN (not a number) value, or some placeholder value that is meant to indicate that this measurement is missing; common examples for the latter are -1 and 0, and this is considered bad practice, even if described in the dataset metadata.

Another type of anomaly are *outliers*. They are values that seem either too large or small to be accurate measurements. An extreme example would be if, for some variable, all the records had values between 0 and 1 except for one record with a value of 42. We can tell something is up there, and most times we'd want to get rid of that record. But where do we draw the line as to what is an outlier and what isn't?

To know help us decide what an outlier is, we first discuss the *interquartile range* (IQR). The IQR is just Q3 - Q1, i.e., the difference between the 75th and 25th percentiles of the data. Note that 50% of the data points are in that span.

A commonly used threshold for outliers is whether a value is lower than Q1 - 1.5 * IQR. Similarly, a value can be considered an outlier if it is greater than Q3 + 1.5 * IQR. The choice for the outlier threshold is somewhat arbitrary; this is just a common choice.

In the next sections, we discuss how to deal with some anomalies, as well as some other useful transformations to data before we actually use it to train a model.

## Data Preprocessing

Once we get to know about our data, what should we do? After profiling our data, there may be a few things we'd like to do with them so that they are more convenient to work with. This is known as *data preprocessing*, where we usually:
- Deal with anomalies in the data.
- Perform feature engineering.
- Apply some otherwise useful modifications to the data, e.g.:
    + Discretization (which we've seen above)
    + Dummification
    + Data balancing
    + Feature scaling and normalization

##### Dealing with Anomalies

The most straightforward method to deal with records that have missing values or that we consider to be outliers is to simply remove them. However, if we have few records and there is a large enough number of missing values for a given feature, we may remove the feature altogether. This choice relates to the curse of dimensionality: there is often a tradeoff between how many records we have and how many features we use to describe them.

Alternatively, we may consider filling in the missing values with something that makes sense. This is called *missing value imputation*. We usually use the mean, the mode, the median, or simply some constant (e.g., NaN) for a given feature to fill its missing values.

We have seen that we usually consider that, for a given feature, a value lower than Q1 - 1.5 * IQR or greater than Q3 + 1.5 * IQR is an outlier, though the choice for this threshold is somewhat arbitrary. The rationale for outliers is similar to the one for missing values: we take into account the dimensionality of the data.

##### Feature Engineering

Domain knowledge, and insights and properties about the data can be used to transform it at the feature level.

The simplest example of feature engineering is to remove variables that are highly correlated. For example, if two variables have a correlation coefficient of 0.99, we can reasonably say that there is redundancy afoot, and so we can drop one of those variables. This is a case of *feature selection*. Another example would be the selection of relevant data features by experts in the respective domain of the problem (e.g., a doctor for medical data).

Another big part of feature engineering is *feature extraction*. In "normal" tabular data, we usually perform dimensionality reduction techniques such as *Principal Component Analysis* (PCA), where we represent data with fewer features and minimal information loss. We can also use neural network-based models such as autoencoders as feature extractors for tabular data, but they generally see greater usage in more complex data such as images, text, and audio.

##### Dummification

We haven't yet looked into any particular statistical/machine learning models, but let us take a linear regressor, for example. Let there also be some dataset where there is some nominal variable for hair color, which may take the values of "black", "brown", "blonde", "red", or "other".

Most models, our linear regressor included, have no idea what "black" or "brown" means; they need numbers in order to perform calculations. As such, we can perform *dummification*, where we take some categorical variable in which the categories are mutually exclusive (such as our nominal variable above) and transform it into *dummy variables*. Specifically, we take each possible value for the nominal feature, and turn it into a separate boolean variable that describes whether each sample has that feature or not. For example, if a record corresponds to a person with blonde hair, then the variable that corresponds to hair being blonde will take the value of 1, and the variables for all other possible hair colors will be 0.

##### Data Balancing

An imbalanced dataset is one in which the number of records for each class is not balanced, i.e., the number of samples from each class varies to a significant degree. For example, if we are trying to train a classifier to tell whether something in an image is a cat or a dog, then we should strive to have a similar amount of images representing cats and dogs (we'll see why in a bit).

Imbalanced datasets may derive from human error, e.g., sampling bias or measurement error or they may be natural to the context of the problem. For example, let's say we are trying to detect fraudulent financial transactions. Since the vast majority of transactions is perfectly normal, we may end up with a dataset where there are hundreds, thousands, or even millions of records representing normal transactions for each fraudulent one.

This bears relevance to how we choose to evaluate our model. For example, if we have 999 records of valid transactions and 1 record of a fraudulent transaction, and our model says that *all* of them are valid, we got an accuracy of 99.9%; but we have failed at our task. In a later topic, we will address many useful ways an perspectives through which to evaluate our models.

Furthermore, models that see a much greater number of samples for a given class may become biased towards classifying more samples as belonging to that class. For example, most neural networks optimize (indirectly, using entropy - but you don't need to know this) for accuracy, which we've seen doesn't tell the whole story about a model's performance. I hope by now you're convinced that imbalanced datasets are a pain, and that (grabs pitchfork) we should definitely do something about it. And indeed, there are a few things we might do:
- Subsampling: if we have a large enough total amount of samples, we may consider simply dropping samples from the majority classes. Obviously, this leads to loss of information.
- Oversampling: simply duplicating samples from the minority classes at random. This incurs a risk of the model overfitting to there more uncommon samples.
- Data augmentation: *technically*, you could say this is oversampling. However, techniques like *SMOTE* (Synthetic Minority Oversampling Technique) and other methods that are informed by the observed data distribution are more sophisticated ways to generate synthetic samples and that really shouldn't be grouped in with "blind" record duplication.

##### Feature Scaling and Normalization

Some machine learning models like *support vector machines* (SVM), *k-nearest neighbors* (KNN), and many clustering algorithms are directly impacted by the scale (i.e., the *range*) of each variable. In particular, they use *distances* (e.g., the Euclidean distance) between data points; so if a feature has a much greater range of values than the others, it will contribute disproportionately to that model's predictive power. To address this, we usually scale the features to the same range.

The most straightforward way to scale a feature to a given range is known simply as *rescaling*, or *min-max normalization*. We usually scale features to the $[0, 1]$ range, and the formula for that, for some feature $X$, is given below:

$$
X' = \frac{X - min(X)}{max(X) - min(X)}.
$$

Could you change this formula to scale the given feature to a range of choice, and not necessarily $[0, 1]$?

The two other more common forms of scaling are *mean normalization*, and *standardization* (or *z-score normalization*). The former simply "zero-centers the mean", i.e., changes the distribution of the variable such that the mean is zero. Standardization does the same thing, but further divides by the standard deviation instead of the range of the feature; this has the effect of making the data have unit variance. Formulas for mean normalization and standardization of a given variable $X$ with mean and standard deviation of $\mu, \sigma$, respectively, are given below:

$$
X' = \frac{X - \mu}{max(X) - min(X)},\quad X' = \frac{X - \mu}{\sigma}
$$

Rescaling is almost a must for similarity-based models, such as KNN and SVMs, and normalization is used in many contexts, such as in many neural network architectures.

## Conclusion

We have briefly discussed data profiling, where we look for interesting aspects of the data. Exploring data is a necessary step in data science, and yields insights about our features and whether and how we should transform them.

Indeed, some transformations can be very helpful (e.g., normalizing the input of neural network layers) and may be outright necessary in certain contexts (e.g., dealing with missing values). Advantages of some of these processes (when applicable) are improving model performance, more efficient training, and dealing with the curse of dimensionality.
