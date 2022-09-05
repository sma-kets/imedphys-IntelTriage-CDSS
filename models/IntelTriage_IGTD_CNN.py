#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import re
from IntelTriage_hyperparameter_optimization import *

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, classification_report, accuracy_score, precision_recall_fscore_support, precision_recall_curve

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Dropout, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adam, Adamax, Nadam

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[2]:


# LOAD DATASET
data = pd.read_csv(r'C:\Users\sma_k\Desktop\USB Drive\IGTD-main\Scripts2\AHEPA_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Αποτέλεσμα'], axis=1), data['Αποτέλεσμα'], test_size = 0.2, random_state = 7)

train_path = r'C:\Users\sma_k\Desktop\USB Drive\IGTD-main\Results\Train_2\data' 
test_path = r'C:\Users\sma_k\Desktop\USB Drive\IGTD-main\Results\Test_2\data' 
train_npy_file_names = os.listdir(train_path)
test_npy_file_names = os.listdir(test_path)


# In[3]:


# functions


# In[3]:


# natural sorting
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, weight_constraint=0,activation='relu', CNN_filters = 4, optimizer='adam',units=32,momentum=0.0, learn_rate=0.001):
    # create model
    model = Sequential()
    model.add(Conv2D(CNN_filters, (2, 2), input_shape=(6,6,1), padding='valid',activation=activation))
    model.add(Conv2D(CNN_filters, (2, 2), padding='valid',activation=activation))
    model.add(Conv2D(CNN_filters, (2, 2), padding='valid',activation=activation))

    #models.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[4]:


train_npy_file_names.sort(key=natural_keys)
test_npy_file_names.sort(key=natural_keys)


# In[5]:


y_train = y_train.sort_index()
y_test = y_test.sort_index()


# In[6]:


#  READ .npy DATA  #
####################
training_data = np.array([np.load(train_path+'\\'+name) for name in train_npy_file_names])
training_data/= 255
training_data = training_data.reshape(training_data.shape[0], training_data.shape[1], training_data.shape[2], 1)

testing_data = np.array([np.load(test_path+'\\'+name) for name in test_npy_file_names])
testing_data/= 255
testing_data = testing_data.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2], 1)

training_data.shape, testing_data.shape


# In[7]:


# define the path to our output directory
OUTPUT_PATH = "output"
# initialize the input shape and number of classes
INPUT_SHAPE = (6, 6, 1)
NUM_CLASSES = 2

# define the total number of epochs to train, batch size, and the
# early stopping patience
EPOCHS = 100
BS = 4
EARLY_STOPPING_PATIENCE = 10


# In[16]:


cv = StratifiedKFold(n_splits=10)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# split into input (X) and output (Y) variables
X = training_data
print(X.shape)
Y = y_train


scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

#defime metrics
scoring = {'AUC': 'roc_auc', 
           'Accuracy': make_scorer(accuracy_score), 
           'Precision': make_scorer(precision_score), 
           'Recall': make_scorer(recall_score)
          }

refit_score = 'Accuracy'

grid = Random_Search_Hyperparameter_Optimization(X, Y, 'CNN', n_splits = 10, n_iter_search = 100, refit_score = refit_score)
#model,  parameters, n_iter=30 , scoring='accuracy', cv=cv
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[17]:


mean_test_keys = [key for key in grid_result.cv_results_.keys() if 'mean_test' in key]
std_test_keys = [key for key in grid_result.cv_results_.keys() if 'std_test' in key]
mean_cv_metrics = pd.DataFrame(index = ['IGTD'], columns = [x.split('_')[2] for x in mean_test_keys])
std_cv_metrics = pd.DataFrame(index = ['IGTD'], columns = [x.split('_')[2] for x in std_test_keys])


result = grid_result
ind_max = np.argmax(result.cv_results_['mean_test_Accuracy'])

for j, key in enumerate(mean_test_keys):
    mean_cv_metrics.loc['IGTD', key.split('_')[2]] = result.cv_results_[key][ind_max]
    std_cv_metrics.loc['IGTD', std_test_keys[j].split('_')[2]] = result.cv_results_[std_test_keys[j]][ind_max]


# In[19]:


mean_cv_metrics


# In[20]:


std_cv_metrics


# In[29]:


grid_result.best_params_


# In[39]:


model = create_model(grid_result.best_params_)
print(model.summary())
model.fit(training_data, y_train, epochs=100, verbose=0)
y_pred = model.predict(testing_data)
predictions = [round(value[0]) for value in y_pred]
# evaluate predictions
print(confusion_matrix(y_test, predictions))
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predictions)
roc_auc = roc_auc_score(y_test, y_pred)
# calculate precision and recall for each threshold
lr_precision, lr_recall, thresholds = precision_recall_curve(y_test, y_pred)
lr_auc = auc(lr_recall, lr_precision)

print(key)
plt.figure(figsize=(5, 5))
plt.title("Precision and Recall Scores as a function of the decision threshold")
plt.plot(thresholds, lr_precision[:-1], "b--", label="Precision")
plt.plot(thresholds, lr_recall[:-1], "g-", label="Recall")
plt.ylabel("Score")
plt.xlabel("Decision Threshold")
plt.legend(loc='best')
plt.show()

probabilities = (y_pred); accuracies = (accuracy_score(y_test, predictions))
recalls = (recall[1]); specificities = (recall[0])
precisions = (precision[1]); fscores = (fscore[1])
roc_aus = (roc_auc); prec_recall_aus = (lr_auc)

result = [accuracies,recalls, specificities, 
           precisions, fscores, roc_aus, 
           prec_recall_aus]

result


# In[ ]:




