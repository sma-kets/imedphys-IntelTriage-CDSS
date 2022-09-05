#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ------------------------ IMPORTS -------------------------------


# In[2]:


from IntelTriage_data_preprocessing import *
from IntelTriage_feature_selection import *
from IntelTriage_hyperparameter_optimization import *
from IntelTriage_classification_metrics import *

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier

import pandas as pd
import numpy

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from sklearn.model_selection import RepeatedKFold


# In[3]:


# -------------------------- FILENAMES AND PATHS --------------------


# In[4]:


AHEPA_dataset_path = r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_\AHEPA_symptoms_unpacked.csv'
public_dataset_path = r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_\public_dataset.csv'

selected_AHEPA_cols = ['U71: Κυστίτιδα / ουρολοίμωξη άλλη', 'P76: Καταθλιπτική διαταραχή',
       'K75: Οξύ έμφραγμα μυοκαρδίου', 'A06: Λιποθυμία', 'R05: Βήχας',
       'K05: Άλλες ανωμαλίες ρυθμού', 'D16: Αιμορραγία ορθού',
       'D01: Γενικευμένο κοιλιακό άλγος', 'T87: Υπογλυκαιμία',
       'N99: Νευρολογική πάθηση, άλλη', 'S99: Νόσος δέρματος, άλλη',
       'R04: Άλλα προβλήματα αναπνοής', 'P19: Χρήση ναρκωτικών',
       'A04: Γενική αδυναμία', 'S12: Δήγμα εντόμου', 'A03: Πυρετός',
       'Φύλο ασθενή', 'D11: Διάρροια', 'K85: Υψηλή αρτηριακή πίεση',
       'S04: Τοπική διόγκωση', 'L02: Συμπτώματα ράχης', 'U06: Αιματουρία',
       'D10: Έμετος', 'K99: Νόσος καρδιαγγειακού, άλλη',
       'R02: Ταχύπνοια / δύσπνοια', 'K04: Αίσθημα παλμών',
       'Καρδιακοί παλμοί (HR) (min)', 'Συστολική αρτηριακή πίεση (mmHg)',
       'Αριθμός νοσηλειών ή επισκέψεων (κατά τον περασμένο χρόνο)',
       'N17: Ίλιγγος / ζάλη', 'Αναπνευστικός ρυθμός (min)',
       'Διαστολική αρτηριακή πίεση (mmg)', 'Κορεσμός οξυγόνου (SpO2) (%)',
       'Κατηγορία άφιξης ασθενή', 'Θερμοκρασία (C)', 'Ηλικία ασθενή',
       'Αποτέλεσμα']


# In[5]:


# functions


# In[6]:


METRICS = [ 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      #keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
      keras.layers.Dense(
          64, activation='relu',
          input_shape=(X_train.shape[-1],)),
      keras.layers.Dense(
          32, activation='relu'),
      keras.layers.Dense(
          64, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])


    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-5),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

    return model


# In[7]:


# -------------------------- LOAD DATASETS --------------------------


# In[12]:


public_dataset = pd.read_csv(public_dataset_path)
AHEPA_dataset = pd.read_csv(AHEPA_dataset_path) 


# In[13]:


# -------------------------- PREPROCESSING --------------------------


# In[14]:


# select ESI > 2, urgent and non urgent incidents
public_dataset = public_dataset[public_dataset['esi'] > 1]

# reduce dimension
public_dataset = public_dataset[selected_AHEPA_cols]
AHEPA_dataset = AHEPA_dataset[selected_AHEPA_cols]

# preprocess
public_dataset = preprocessing(public_dataset, 'public_dataset', discret_list = [], norm=False)
AHEPA_dataset = preprocessing(AHEPA_dataset, 'AHEPA_dataset', discret_list = [], norm=False)


# In[16]:


ones = public_dataset[public_dataset['Αποτέλεσμα']==1].shape[0]
public_dataset = pd.concat((public_dataset[public_dataset['Αποτέλεσμα']==0][:ones], public_dataset[public_dataset['Αποτέλεσμα']==1]))
public_dataset[public_dataset['Αποτέλεσμα']==1].shape[0], public_dataset[public_dataset['Αποτέλεσμα']==0].shape[0], public_dataset.shape[0]


# In[22]:


# Transfer learning


# In[134]:


# ----------------------------------------TRAIN - TEST split--------------------------
X_train, X_test, y_train, y_test = train_test_split(public_dataset.drop('Αποτέλεσμα',axis=1), public_dataset['Αποτέλεσμα'], test_size = 0.1, random_state=7)


# In[135]:


EPOCHS = 50
BATCH_SIZE = 8

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=0,
    patience=30,
    mode='max',
    restore_best_weights=True)


# In[136]:


model = make_model()
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)
model.summary()


# In[137]:


model = make_model()
#model.load_weights(initial_weights)
baseline_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_split=0.2,
    verbose=1)


# In[23]:


# added 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold

y_pred_probs = model.predict(X_test)    
y_pred = [0 if pred<0.5 else 1 for pred in y_pred_probs]
predictions = [round(value) for value in y_pred]
# evaluate predictions
print(confusion_matrix(y_test, y_pred))
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_probs)
# calculate precision and recall for each threshold
lr_precision, lr_recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
lr_auc = auc(lr_recall, lr_precision)

plt.figure(figsize=(5, 5))
plt.title("Precision and Recall Scores as a function of the decision threshold")
plt.plot(thresholds, lr_precision[:-1], "b--", label="Precision")
plt.plot(thresholds, lr_recall[:-1], "g-", label="Recall")
plt.ylabel("Score")
plt.xlabel("Decision Threshold")
plt.legend(loc='best')
plt.show()
#append values for each fold
#print(classification_report(y_test, predictions))
probabilities = []; accuracies = [];
precisions = [] ; specificities = []; recalls = [] ;
roc_aus = [] ; prec_recall_aus = [] ; probabilities = [] ; fscores = [] ; 

probabilities.append(y_pred_probs); accuracies.append(accuracy_score(y_test, predictions));
recalls.append(recall[1]); specificities.append(recall[0]); precisions.append(precision[1]);
fscores.append(fscore[1]); roc_aus.append(roc_auc); prec_recall_aus.append(lr_auc);

results = {'accuracy': accuracies,
'recall': recalls,
'specificity': specificities,
'precision': precisions,
'fscore': fscores,
'roc_auc': roc_aus,
'prec_recall_auc': prec_recall_aus}

results = pd.DataFrame(results)

results


# In[138]:


# ----------------------------------------TRAIN - TEST split--------------------------
X_train1, X_test1, y_train1, y_test1 = train_test_split(AHEPA_dataset.drop('Αποτέλεσμα',axis=1), AHEPA_dataset['Αποτέλεσμα'], 
                                                        test_size = 0.2, random_state=7)


# In[139]:


# test on AHEPA without tuning
model = keras.models.load_model(r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_')
probabilities = []; accuracies = [];
precisions = [] ; specificities = []; recalls = [] ;
roc_aus = [] ; prec_recall_aus = [] ; probabilities = [] ; fscores = [] ; 

y_pred_probs = model.predict(X_test1)    

classification_metrics(y_test1, y_pred_probs, prob_threshold = 0.5, plot_ConfusionM = False)


# In[47]:


# save the model to disk
#model.save(r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_')


# In[143]:


# TEST STACKING utilizing the tuned models
# 10 fold cross validation
splits = 10
transfer_learning_10fold = pd.DataFrame(columns = ['accuracy', 'precision', 'specificity', 'recall', 'AU-ROC', 'precision-recall AU'])
skf = StratifiedKFold(n_splits=splits)

import gc

for n, indices in enumerate(skf.split(X_train1, y_train1)):
    
    train_index, test_index = indices
    X_train, X_test = X_train1.iloc[train_index], X_train1.iloc[test_index]
    y_train, y_test = y_train1.iloc[train_index], y_train1.iloc[test_index]
    
    # load model
    model = keras.models.load_model(r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_')
    
    model.layers[0].trainable = False
    model.layers[1].trainable = False

    model.fit( X_train,  y_train,
        batch_size=BATCH_SIZE,
        epochs=500, verbose=0) #, callbacks=[early_stopping],
    #validation_split=0.1)
    
    # make predictions
    y_pred_probs = model.predict(X_test)   
    
    transfer_learning_10fold.loc[str(n)] = classification_metrics(y_test, y_pred_probs, prob_threshold = 0.5, plot_ConfusionM = False).iloc[0].values
    
    del model
    gc.collect


# In[145]:


transfer_learning_10fold.mean()


# In[146]:


model = keras.models.load_model(r'C:\Users\sma_k\my_python_scripts_etc\IntelTriage_')

print(len(model.layers))
model.layers[0].trainable = False
model.layers[1].trainable = False

print("weights:", len(model.weights))
print("trainable_weights:", len(model.trainable_weights))
print("non_trainable_weights:", len(model.non_trainable_weights))

model.fit(
    X_train1,
    y_train1,
    batch_size=4,
    epochs=500,verbose=0)
    #, callbacks=[early_stopping],
    #validation_split=0.2)


# In[147]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import RepeatedKFold

probabilities = []; accuracies = []; precisions = [] ; specificities = []; recalls = [] ; roc_aus = [] ; prec_recall_aus = [] ; probabilities = [] ; fscores = [] ;

y_pred = model.predict(X_test)

classification_metrics(y_test, y_pred_probs, prob_threshold = 0.5, plot_ConfusionM = True).iloc[0]


# In[ ]:




