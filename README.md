# machine-learning


## 1. ML Theory:

## 2. ML with Python


## 1. ML Theory:

### What is Machine Learning

**What is Machine Learning:**

Find patterns in data. Use those patterns to predict the future.

**What does it mean to Learn:**

**Machine Learning in a Nutshell:**

1. You find data that contains patterns
2. Feed that data into machine learning algorithms to find patterns in the data
3. That algorithm generate something as known - Model. A model - functionality that can recognize patterns with new data.
4. Application - supplies new data to model to see if it matches known patterns. 

Please see the example ![diagram1](./pictures/ml-in-nutshell.png)

**What is a data scientist:**

Someone who knows about:

- Statistics

- Machine learning software

- Some problem domain(ideally)

### The Machine Learning Process is:

The First Problem: **Asking the Right Question**

- Ask yourself: Do you have the right data to answer this question?

- Do you know how you'll measure success?

- Model: how good those predictions must be to make this process qualify to make it success?


The Machine Learning process:

1. To start you choose that data(what data is more likely to be predictive)

2. Prepared data to be ready for processing, basically apply pre-processing to data (clean, organize, format, get it prepared)

3. Applying learning algorithms to that prepared data.

4. Create candidate Model -  iterate to find the best model

5. Deploy chosen model

Please see the example ![diagram2](./pictures/ml-process.png)


The next step is repeating the Machine Learning process regulary:


Please see the example ![diagram3](./pictures/repeating-ml-process.png)


### Machine-Learning Concepts:

**Terminology:**

*Training data* - prepared data to use to create Model

*Supervised learning* - the value you want to predict is in the training data. The data is labeled.

*Unsupervised learning* - the value you want to predict is not in the training data. The data is unlabeled.


### Data Pre-processing with Supervised Learning:

Training-data, please see the example ![diagram4](./pictures/training-data.png)


**Categorizing Machine Learning Problems: Regression**

please see the example ![diagram5](./pictures/regression.png)


**Categorizing Machine Learning Problems: Classification**

please see the example ![diagram6](./pictures/classification.png)


**Categorizing Machine Learning Problems: Clustering**

please see the example ![diagram7](./pictures/clustering.png)


**Styles of Machine Learning Algorithms:**

please see the example ![diagram8](./pictures/ml-algorithms.png)

**Training a Model with Supervised Learning: Choose Features**

please see the example ![diagram9](./pictures/training-with-supervised.png)


**Testing a Model with Supervised Learning: Test the results and compare target values generated from test data  with actual target values**

please see the example ![diagram10](./pictures/test-with-supervised.png)


**Improve a Model with Supervised Learning: Some Options**

please see the example ![diagram11](./pictures/improve-with-supervised.png)


**Using a Model:**

please see the example ![diagram12](./pictures/using-model.png)



## 2. ML with Python

**Machine Learning Logic**

1. Data - get data and modify into format that ML can use

2. Algorithm - pass these data into algorithm

3. Data Analysis - analyses these data and create a model

4. Model - solution to solve the problem based on input data

**Machine Learning Technique Comparison:**

please see the example ![diagram13](./pictures/ml-comparison.png)


### Machine Learning Workflow:

MLW - an orchestrated and repeatable pattern which systematically transforms and processes information to create prediction solutions.


1. Asking the right question(goals we want to achieve, data we need and process we want to perform)

2. Preparing data (gather the data we need to answer our questions)

3. Selecting the algorithm (consider which algorithm we use)

4. Training the model

5. Testing the model (test accuracy, generate statistics)


#### Asking the right question:



Example: "Predict if a person will develop diabetes"

Need statement to direct and validate work:

 - Define scope(including data source):
  
Predict if a person will be develop diabetes;

Identify critical features;

Focus on at risk population

Select data source

Pima Indian Diabetes study is a good source

Summary: Using Pima Indian Diabetes data, predict which people will develop diabetes


 - Define target performance(what prediction accuracy we should expect)
 

We will define diabetes in Binary fashion: diabetes or not diabetes

Binary result (True or False)

Genetic difference are a factor

70% Accuracy is common target

Summary:  Using Pima Indian Diabetes data, predict with 70% or great accuracy. which people will develop diabetes


 - Define context of usage
 
 
 What does it mean disease prediction
 
 Medical research practices
 
 Unknown variations between people
 
 Likelihood is used
 
 Summary:  Using Pima Indian Diabetes data, predict with 70%, which people are likely to develop diabetes

 
 - Define how solution will be created
 
 Machine Learning Workflow:
 
 Process Pima Indian data
 
 Transform data as required
 
 Summary:  Use the Machine Learning Workflow to process and transform Pima Indian data to create a prediction model. 
 This Model must predict with people are likely to develop diabetes with  70% or greater accuracy.
 
 
 
#### 2. Preparing Data

80% of work are getting, cleaning and organize the data

Data Rule #1: closer the data is to what you are predicting, the better

Data Rule #2: Data will never be in the format you need

Please see the following link with Notebook example: https://github.com/constantine77/machine-learning/blob/master/notebooks/pima-prediction-diabetes.ipynb

#### 3. Selecting the Algorithm

How to decide which algorithm to use?

Algorithm Selection:

1. Learning Type (what learning type they support)

2. Result (result type the algorithm predicts)

3. Complexity (the complexity of the algorithm)

4. Basic vs enhanced


1. Learning Type - we are looking into solution statement and guidance it offered.
We see the prediction keyword - prediction means supervised machine learning.

2. Result Type - prediction can be divided into two subcategories: Regression and Classification.

Regression means: continuous set of values

Classification means: discrete values, small, medium, large or true and false. 

3. Complexity: keep it simple and eliminate "ensemble" algorithms.
 
4. Basic vs enhanced: we choose basic algorithm

Candidate Algorithms:

Naive Bayes

Logistic Regression

Decision Tree















## Installation:

**Python:**

numpy - scientific computing

```buildoutcfg
pip install numpy
```


pandas - data frames

```buildoutcfg
pip3 install pandas
```

matplotlib - 2d plotting

```buildoutcfg
pip3 install matplotlib
```

scikit-learn - ML algorithms, pre-processing, performance evaluation, etc...

```buildoutcfg
python3 -m pip install scikit-learn
```

**Jupyter Notebook**

command to lunch:

```
jupyter notebook
```

Tips:

shift + tab - you can see command description






## Links:

1. Understanding Machine Learning:
https://app.pluralsight.com/player?course=understanding-machine-learning&author=david-chappell&name=understanding-machine-learning-m4&clip=6&mode=live

2: Understanding ML with Python:

