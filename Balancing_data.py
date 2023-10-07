

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


from imblearn.under_sampling import (
    RandomUnderSampler,
    CondensedNearestNeighbour,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    TomekLinks
)

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE
)

import pandas as pd
from google.colab import drive

# Mount Google Drive to access the file
drive.mount('/content/drive')

# Read the CSV file
data = pd.read_csv('/content/drive/MyDrive/credit_customers.csv')

# Check the first few rows of the DataFrame
data = data.drop_duplicates()
data.drop(['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'], axis=1, inplace=True)

data['class'] = data['class'].replace({'good': -1})
data['class'] = data['class'].replace({'bad': 1})

# Rename the 'Class' column to 'Target'
data = data.rename(columns={'class': 'target'})

data.head()



data.shape
data.columns

data.target.value_counts() / len(data)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['target'], axis=1),
    data['target'],
    test_size=0.3,
    random_state=0
)
X_train.shape, X_test.shape

rus = RandomUnderSampler(
      sampling_strategy='auto',
      random_state=0,
      replacement=True
    )

X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

cnn = CondensedNearestNeighbour(
    sampling_strategy='auto',
    random_state=0,
    n_neighbors=1,
    n_jobs=4
)

X_resampled, y_resampled = cnn.fit_resample(X_train, y_train)

tl = TomekLinks(
    sampling_strategy='auto',
    n_jobs=4
)
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)

enn = EditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    kind_sel='all',
    n_jobs=4)

X_resampled, y_resampled = enn.fit_resample(X_train, y_train)

renn = RepeatedEditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    kind_sel='all',
    n_jobs=4,
    max_iter=100
)
X_resampled, y_resampled = renn.fit_resample(X_train, y_train)

oss = OneSidedSelection(
    sampling_strategy='auto',
    random_state=0,
    n_neighbors=1,
    n_jobs=4
)
X_resampled, y_resampled = oss.fit_resample(X_train, y_train)

allKnn = AllKNN(
    sampling_strategy='auto',
    n_neighbors=5,
    kind_sel='all',
    n_jobs=4
)
X_resampled, y_resampled = allKnn.fit_resample(X_train, y_train)

ros = RandomOverSampler(
    sampling_strategy='auto',
    random_state=0
)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

shrink = 2  #change to see results diffrence

ros = RandomOverSampler(
    sampling_strategy='auto',
    random_state=0,
    shrinkage=shrink
)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

sm = SMOTE(
    sampling_strategy='auto',
    random_state=0,
    k_neighbors=5
)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

[8]
X_resampled.shape, y_resampled.shape

[9]
y_train.value_counts()

[10]
y_resampled.value_counts()

def run_randomForests(X_train, X_test, y_train, y_test):

  rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
  rf.fit(X_train, y_train)

  print('Train set')
  pred = rf.predict_proba(X_train)
  print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

  print('Test set')
  pred = rf.predict_proba(X_test)
  print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

run_randomForests(X_train,
                    X_test,
                    y_train,
                    y_test)

run_randomForests(X_resampled,
                    X_test,
                    y_resampled,
                    y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def run_randomForests(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
    rf.fit(X_train, y_train)

    pred_train = rf.predict_proba(X_train)
    pred_test = rf.predict_proba(X_test)

    # Predict the class labels
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Calculate the confusion matrices
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    return cm_train, cm_test

# Example usage
cm_train, cm_test = run_randomForests(X_resampled, X_test, y_resampled, y_test)

# Plot the confusion matrix for the test set
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[-1, 1])
disp.plot()
plt.show()