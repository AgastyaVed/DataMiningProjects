
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
from sklearn.utils import resample
# import featuretools as ft


# ## Read the data

# In[5]:


d = pd.read_excel('default of credit card clients.xls', header=1)


# In[6]:


d.set_index('ID')


# ## EDA

# In[31]:


def data_desc(d):
    print('the no. of columsn are: ',len(d.columns))
    print('The distinct column types are: ', )
    for i in d.columns:
        print('The column ',i, 'is of ',d.loc[:,i].dtype)
data_desc(d)


# Essential stats of the data

# In[26]:


d.describe()


# #### Finding out if columns are correlated

# Visualization: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

# In[21]:


# plt.figure(figsize=(8, 6))
# d.hist()
# plt.figure(figsize=(8, 6))
# plt.show()


# Correlation matrix plot: https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166

# In[27]:


#sns.pairplot(d)


# In[88]:


corr = d.corr()
print(type(corr))


# ### Missing values

# In[84]:


def missing_values(d):
    null_cols = d.loc[:,d.isnull().sum() != 0].isnull().sum()
    if t1.empty:
        print('There are no null columns in the data')
    else:
        print(null_cols.sort_values(axis=0))
missing_values(d)


# ### Finding if categorical columns are there

# To find out the number of unique columsn in the data for each of the columns I used list comprehension. <br>
# List comprehension in python: https://caisbalderas.com/blog/iterating-with-python-lambdas/ 

# In[57]:


[[i, d[i].nunique()] for i in d.columns]


# In[70]:


def find_cat_cols(d, cutoff_criteria):
    '''
    Returns the column names with number of unique values less than the cutoff criteria number
    '''    
    uniq_col_count = d.nunique()    
    cat_vars = uniq_col_count[uniq_col_count<cutoff_criteria]
    print('The columns less than', cutoff_criteria, 'features are: \n', cat_vars)


# In[72]:


find_cat_cols(d,10)


# The categorical columns are: SEX, EDUCATION, MARRIAGE. From data description the columns with names prefix 'PAY_' are also categorical

# ### Encoding categorical columns

# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above <br><br>
# **How are numbers assigned to the history of payment columns**<br>
# There are 11 classes of customers. The greater the number, the greater the probability of defaulting
# 

# In the later methods we can add some additional features. For example one feature that I see that can add more information is the total number of delayed months give by the sum of the columsn X6-X11. The higher the number, the higher the chances of defaulting. 

# #### Split the data into training, validation and test dataset

# randomly split the data into training and testing sets, training = 60%, test, validation 20% each

# In[99]:


train, validate, test = np.split(d.sample(frac=1), [int(.6*len(d)), int(.8*len(d))])


# In[101]:


test.head()


# ## Machine Learning

# #### Logistic regression

# Since this is a binary classification problem, Logistic regression is first tested to see how well a simple model performs in this case. We will use L2 regularization. Few notes from sklearn documentation itself about logistic regression: 
# 
#     For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
#     For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
#     ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.
# 

# In[ ]:


def log_reg_model(data, C):
    X = data.loc[:,data.columns != 'default payment next month']
    y = data.loc[:,data.columns == 'default payment next month']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    


# pandas and sklearn have interaction problems. pandas series is also a 2D array with the first coulmn as index. in sklearn models such as logistic regression it needs the input array in the shape (n_samples, ). So the pandas series has to be converted into this shape

# In[150]:


data = train
X = data.loc[:,data.columns != 'default payment next month'].values
y = data.loc[:,data.columns == 'default payment next month'].values.reshape(len(y.values),)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)


# In[145]:


for i, C in enumerate((1, 0.1, 0.01)):
    # turn down tolerance for short training time
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga')
    clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver='saga')
    clf_l1_LR.fit(X_train, y_train)
    clf_l2_LR.fit(X_train, y_train)
    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()
#     fpr_l1_LR, tpr_l1_LR = roc_curve(y_test, y_pred_l1_LR)
#     fpr_l2_LR, tpr_l2_LR = roc_curve(y_test, y_pred_l2_LR)
#     y_pred_l1_LR = clf_l1_LR.predict(X_test)
#     y_pred_l2_LR =  clf_l2_LR.predict(X_test)    
    print("score with L1 penalty: %.4f" % clf_l1_LR.score(X_test, y_test))    
    print("score with L2 penalty: %.4f" % clf_l2_LR.score(X_test, y_test))
    
#     # Plotting
#     l1_plot = plt.subplot(3, 2, 2 * i + 1)
#     l2_plot = plt.subplot(3, 2, 2 * (i + 1))
#     if i == 0:
#         l1_plot.set_title("L1 penalty")
#         l2_plot.set_title("L2 penalty")

# #     l1_plot.imshow(np.abs(coef_l1_LR.reshape(8, 8)), interpolation='nearest',
# #                    cmap='binary', vmax=1, vmin=0)
# #     l2_plot.imshow(np.abs(coef_l2_LR.reshape(8, 8)), interpolation='nearest',
# #                    cmap='binary', vmax=1, vmin=0)
# #     plt.text(-8, 3, "C = %.2f" % C)

#     l1_plot.set_xticks(())
#     l1_plot.set_yticks(())
#     l2_plot.set_xticks(())
#     l2_plot.set_yticks(())
    


# #### SVM classifier

# Scaling of the data is important in SVM. More here: https://neerajkumar.org/writings/svm/ <br>
# Scikit learn: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

# If the data has to be scaled then how do we handle categorical data?
# https://stats.stackexchange.com/questions/52915/how-to-deal-with-an-svm-with-categorical-attributes
# https://sebastianraschka.com/faq/docs/svm_for_categorical_data.html
# This says that if the categorical variables are ordinals then the data is good for SVM

# In[7]:


data = d.copy()
X = data.loc[:,data.columns != 'default payment next month'].values
y = data.loc[:,data.columns == 'default payment next month'].values
y = y.reshape(len(y),)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)


# Let's fit a radial basis function SVM and then use balanced data to see how the performance improves.

# In[10]:


clf = svm.SVC(kernel='rbf',gamma=10)


# In[11]:


clf.fit(X_train, y_train)


# In[12]:


y_pred = clf.predict(X_test)


# Metrics for SVM: https://scikit-learn.org/stable/modules/model_evaluation.html

# In[16]:


metrics.accuracy_score(y_test, y_pred)


# ### How to improve the fit

# Observed that there are few things that I can explore further to develop my understanding of SVM further: <br>
# 1) How does the target data being unbalanced affect the classification? https://elitedatascience.com/imbalanced-classes
# https://stats.stackexchange.com/questions/94295/svm-for-unbalanced-data <br>
# 2) How does standard scaling of the data affect the classification? https://chrisalbon.com/machine_learning/support_vector_machines/imbalanced_classes_in_svm/ <br>
# 3) Hyper-parameter tuning - Which Kernel and what values of C and gamma <br>
# 4) What metrics should I use to judge the goodness of the model? <br><br>
# A very good explanation of the various metrics and their connections with SVM: https://stats.stackexchange.com/questions/73537/choosing-a-classification-performance-metric-for-model-selection-feature-select

# #### 1. Balancing the classes

# SVM depends on whether the data is balanced for the classes we're trying to classify. So let's check if the current data is balanced

# In[9]:


data['default payment next month'].value_counts()


# In[9]:


d_major_class = d[d['default payment next month']==0]
d_minor_class = d[d['default payment next month']==1]


# In[10]:


# Downsample major class
d_major_class_downsampled = resample(d_major_class, replace=False, n_samples = d_minor_class.shape[0], random_state = 999)


# In[12]:


# Now concatenate the minor and downsampled major class datasets into one
d_downsampled = pd.concat([d_major_class_downsampled, d_minor_class])

# Value counts of the downsampled dataset
d_downsampled['default payment next month'].value_counts()


# Now we see that the classes are balanced. Now let's fit SVM model to this dataset.

# In[16]:


def svm_classifier(data, test_train_split_ratio, kernel):
    #Splitting the data into training and test sets
    X = data.loc[:,data.columns != 'default payment next month'].values
    y = data.loc[:,data.columns == 'default payment next month'].values
    y = y.reshape(len(y),)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_train_split_ratio, random_state=42)    
    clf = svm.SVC(kernel=kernel,gamma=10)
    clf.fit(X_train, y_train)    
    # Prediciting using the model
    y_pred = clf.predict(X_test)    
    # Metrics
    print("Accuracy using",kernel,"is-",metrics.accuracy_score(y_test, y_pred))

#     #Building the model
#     for kernel in enumerate(('linear', 'rbf', 'poly')):
#         clf = svm.SVC(kernel=kernel, gamma=10)
#         clf.fit(X_train, y_train)    
#         # Prediciting using the model
#         y_pred = clf.predict(X_test)    
#         # Metrics
#         print("For kernel-",kernel,"accuracy is-",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


svm_classifier(d_downsampled, 0.33)


# In[17]:


get_ipython().run_cell_magic('time', '', "svm_classifier(d_downsampled, 0.33, 'rbf')")


# SVM classifier took a long time to run. Here are some discussions on it: https://stackoverflow.com/questions/18165213/how-much-time-does-take-train-svm-classifier<br>
# The factor C, kernel(rbf the most complicated one) and data size are the chief factors. <br>
# I observed that the radial basis function kernel runs much faster compared to the linear kernel.

# #### 2. Standard scaling

# In[8]:


X = StandardScaler().fit_transform(X)


# In[21]:


num_cols = ['LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']


# In[38]:


list(d)


# How to change data type of multiple columns in pandas dataframe: https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas

# In[35]:


t1 = d.copy()
# t1.head()
t2 = t1[num_cols]

t3 = t2.apply(pd.to_numeric)
# t1 = StandardScaler().fit_transform(t1)


# In[37]:


# print(t2.dtypes)
print(t3.dtypes)


# In[ ]:


def svm_classifier(data, test_train_split_ratio, kernel):
    # Standaa
    #Splitting the data into training and test sets
    X = data.loc[:,data.columns != 'default payment next month'].values
    y = data.loc[:,data.columns == 'default payment next month'].values
    y = y.reshape(len(y),)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_train_split_ratio, random_state=42)    
    clf = svm.SVC(kernel=kernel,gamma=10)
    clf.fit(X_train, y_train)    
    # Prediciting using the model
    y_pred = clf.predict(X_test)    
    # Metrics
    print("Accuracy using",kernel,"is-",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()


# Test out different kernel and see which one fits the best

# Some nice articles on feature selection for the Kaggle Credit defaulter dataset, which is much more complex problem as it consists of multiple datasets that need to be joined. 
