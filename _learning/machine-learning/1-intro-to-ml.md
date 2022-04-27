---
title: Introduction to Machine Learning
slug: intro-to-ml
background: '/img/bg-post.jpg'
lesson_number: 1
---

Welcome to the Machine Learning section!

Conventionally, we design algorithms such that, when given some input, they yield desirable output according to some metric. For example, [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) is used to find the shortest (or least costly) paths between nodes in a graph. In machine learning (ML), we design algorithms that make use of data (known as *training data*) in order to improve.

Abstractly, we conventionally develop algorithms that perform a certain task well by correctly mapping inputs to outputs. In most of ML we develop algorithms that, given examples of inputs and (sometimes) outputs, *learn* that mapping between them in order to perform well on a task.

In the next section, we discuss some concepts regarding data - which after all, is the coal that allows the machine learning train to go *choo choo*. After that, we define the main 3 types of machine learning.

## Data - Basic Concepts

In statistics and machine learning, we usually need to perform calculations using information about objects or entities of the real world. For that purpose, we choose a set of variables which we use to describe them. For example, hair color, blood pressure (mm Hg), number of kidneys, and life stage are variables which we can measure in humans.

A *record* corresponds to the observed values for the variables we use to describe an object. In other words, a record is an n-tuple whose elements are facts about something in the real world. In the example given above, we could describe humans by their hair color, blood pressure, number of kidneys, and life stage. In that case, we could have a record $x = $ {"Brown", 114.2, 1, "Adult"} which would correspond to the measurements for a given person.

In most machine learning problems, we deal with a set of records (or *instances*, or *samples*, or *observations*) and their respective *targets*. A target variable of some record is just a variable of interest which we would like to learn, given the other features that describe the data. Indeed, samples are described by features (or *attributes*), which we can use to try to predict or approximate their target value. Since we tend to represent records as multidimensional vectors, the variables of the data can also be referred to as its "dimensions". Variables are either *numerical* or *categorical*.

Numerical variables are quantitative, since they literally quantify something (in this case, about an observation). Numerical variables can either be:
- Discrete: can take only a finite number of real values, resembles "counting" (e.g., number of apples).
- Continuous: can take any real value within a certain range (e.g., height, percentage, ratio).

Categorical variables, also called *symbolic* variables, are qualitative, since they describe something that is not quantifiable. Categorical variables can either be:
- Nominal: some label whose possible values have no order (e.g., binary variables, gender, name).
- Ordinal: some label whose possible values have an order relationship (e.g., describing difficulty as "hard", "medium", "easy").

Let's take the variables we used to describe a human above. If we use color names - such as "green", "blue", "black", etc. - to describe someone's hair color, we have a finite number of labels (names) which we can attribute to an object. Furthermore, there is no natural ordering among colors in this context, and so hair color is a nominal (categorical) variable. Blood pressure can, in theory, take any real value within some arbitrary range, even if in practice we are limited to the precision of whatever representation we are using (e.g., 64-bit floating point numbers on a computer). As such, blood pressure is a continuous (numerical) variable. Humans can only have an integer number of kidneys; we don't have 1.2 or 0.5 kidneys. We either have (for simplicity) either 0, 1, or 2. Henceforth, the number of kidneys a human has is a discrete (numerical) variable. Finally, if we imagine that the possible life stages of a human are "baby", "child", "teen", "adult", and "elderly", then we are dealing with a set of labels with an order relationship with respect to time. Thus, the life stage of a human is an ordinal (categorical) variable.

> Note: you might be wondering whether ordinal variables can be numeric. For example, if you rate something from {1, 2, 3, 4, 5}, then you'll have a number with ordinal meaning. However, we still consider them to be categorical variables.

## Supervised Learning

When we have a dataset whose samples are annotated, i.e., we know the value of the target variable for each sample, then we are dealing with a task of *supervised learning*.

Let us look at the small dataset in the table below:

$$
\begin{array}{| c | c | c | c |}
	\hline
	\textbf{Leaf Color} & \textbf{Trunk Height (m)} & \textbf{Trunk Diameter (cm)} & \textbf{Dead} \\
	\hline
	\text{Brown} & 3.81 & 84.12 & \text{Yes} \\
	\hline
	\text{Brown} & 4.56 & 101.32 & \text{Yes} \\
	\hline
	\text{Green} & 4.21 & 98.71 & \text{No} \\
	\hline
	\text{Green} & 4.06 & 76.68 & \text{Yes} \\
	\hline
\end{array}
$$

Here, we have 4 samples. In this case, we want to guess whether a tree is dead judging by some of its features: leaf color, trunk height, and trunk diameter. This means that, based on those 3 variables that describe each tree, we want to predict the value of its target variable, which is whether it is dead or not.

As discussed above, we usually represent samples of some dataset as multidimensional vectors. As such, we use $\v{x}$ to refer to a sample, and $y$ to refer to its target variable. Furthermore, since a dataset can be seen as a 2D matrix where each row is a sample and each column is a feature that describes those samples, we often use $\v{X}$ to refer to the entire set of the samples, and $\v{y}$ to refer to their corresponding target values.

In practice, it's useful to have all the variables be real numbers. For example, if we assume that the leaf color of some tree can only be either brown or green, then we could use a 0 to indicate that a tree has a brown leaf color, and 1 to indicate that it has a green leaf color. Similarly, for the target variable, we can use 0 and 1 to say whether a tree is dead or alive, respectively. This practice resembles that of [dummification](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)), and we will go over both in a later section.

With this in mind, we can represent the dataset more "mathematically":

$$
\v{X} =
	\begin{bmatrix}
	0 & 3.81 & 84.12 \\
	0 & 4.56 & 101.32 \\
	1 & 4.21 & 98.71 \\
	1 & 4.06 & 76.68
	\end{bmatrix},\
\v{y} =
	\begin{bmatrix}
	1 \\
	1 \\
	0 \\
	1
	\end{bmatrix}
$$

As we have seen above, in this case, we'd like to predict whether a tree (as described by its leaf color, trunk height, and trunk diameter) is dead or not. This is an example of a *classification* task, where we try to predict a categorical variable. In particular, since there are exactly *two* possible classes that a sample may belong to (dead or not), we call this a *binary* classification task, wherein a sample belongs to either one class or the other.

If there are more than two possible classes that a sample may belong to, e.g., predicting the species of a flower given its attributes, then we are dealing with a *multi-class* classification task. In both binary classification and multi-class classification, classes are mutually exclusive; a sample cannot belong to more than one class. In cases where we want to predict multiple class labels for each example, e.g., identifying multiple types of entities (people, vehicles, buildings, etc.) in a photo, we are dealing with a *multi-label* classification task.

Finally, we may not want to predict a categorical (i.e., discrete) variable at all. For example, if instead of wanting to know whether a tree is dead or not, we wanted to predict its age in years, we would be trying to predict a continuous variable. When we try to predict a continuous numerical value (such as the age of something), we are dealing with a *regression* task.

Supervised learning methods include *Naïve Bayes*, *k-nearest neighbors* (KNN), *support vector machines* (SVM), *linear regression*, *neural networks*, and many others.

## Unsupervised Learning

*Unsupervised learning* addresses situations in which we do not have annotated data. In other words, we now do not have target variables provided by an expert. The goal of unsupervised learning techniques is thus to learn some structure or compact internal representation of the data without making use of labels.

One main method of unsupervised learning is that of *clustering*, or *cluster analysis*. We use clustering techniques to group data points into groups (clusters) in order to get a sense of the relationships between them. Examples of clustering methods are *k-means*, DBSCAN, and mixtures (e.g., Gaussian mixture).

Another method of unsupervised learning is *dimensionality reduction*, which can be seen as learning latent variables in the data. Techniques for dimensionality reduction include *expectation-maximization*, *principal components analysis* (PCA), and *singular value decomposition*.

Finally, we can see [density estimation](https://en.wikipedia.org/wiki/Density_estimation) as another main goal of unsupervised learning. For example, while we may think of neural networks as models that perform discriminative tasks, we also use them in unsupervised contexts. For example, *autoencoders* and *generative adversarial networks* are methods that use neural networks in an unsupervised context in order to learn the data distribution and generate fake realistic samples.

> Note: "semi-supervised learning", as the name implies, is somewhere between supervised and unsupervised learning: we have only some labeled data. For now though, let's forget about this particular case.

## Reinforcement Learning

In general, *reinforcement learning* concerns artificially intelligent agents interacting (through actions) with some environment while trying to maximize some reward (or minimize some cost).

Due to the broad applications of the above definition, we see the field of reinforcement learning being mentioned in different disciplines, and even different fields of study entirely: multi-agent and autonomous agent systems (computer science), operations research (industrial engineering), game theory (economics), control theory (electronic engineering), etc.

Reinforcement learning models, usually called *agents*, do not necessarily use labeled data to learn. Instead, they learn by interacting with their environment and being rewarded or punished. However, some approaches (e.g., deep reinforcement learning) combine reinforcement learning algorithms with supervised learning.

## Conclusion

We discussed a high-level definition of machine learning, and saw how it is different from the classical approach in algorithm development. Then, we went over some basic concepts that allow us to better understand data and the problems we address with machine learning. We now also know the three main ways in which machine learning models can learn, and some of their applications.