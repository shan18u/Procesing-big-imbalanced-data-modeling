
!pip install -U scikit-learn
!pip install numpy
!pip install scipy

import numpy as np

from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

# TPR and FPR values for all thresholds
thresholds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.3, 5.5, 6.0, 6.2, 7.0, 8.0]
TPR = [1.0, 1.0, 0.8333, 0.6667, 0.6667, 0.8333, 0.8333, 0.6667, 0.5, 0.3333, 0.1667, 0.0]
FPR = [1.0, 0.75, 0.75, 0.75, 0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]

# Plot the ROC curve
plt.plot(FPR, TPR)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
y_scores = np.array([5.0, 1.0, 3.0, 2.0, 6.0, 4.0, 5.5, 6.2, 7.0, 5.3])
precision,recall, thresholds = precision_recall_curve(y_true, y_scores)
print(precision)
print(recall)

import matplotlib.pyplot as plt

# Dataset 1
fpr1 = [0.000, 0.000, 0.000, 0.000, 0.500, 0.500, 0.500, 0.500, 0.500]
tpr1 = [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
recall1 = [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
precision1 = [1.000, 1.000, 1.000, 1.000, 0.800, 0.833, 0.857, 0.875, 0.889]

# Dataset 2
fpr2 = [0.000, 0.000, 0.200, 0.200, 0.200, 0.400, 0.400, 0.600, 0.800, 1.000]
tpr2 = [0.000, 0.200, 0.200, 0.400, 0.600, 0.600, 0.800, 0.800, 0.800, 0.800]
recall2 = [0.000, 0.200, 0.200, 0.400, 0.600, 0.600, 0.800, 0.800, 0.800, 0.800]
precision2 = [1.000, 1.000, 0.500, 0.667, 0.750, 0.600, 0.667, 0.571, 0.500, 0.444]

# Plot ROC curves
plt.figure()
plt.plot(fpr1, tpr1, label="Dataset 1")
plt.plot(fpr2, tpr2, label="Dataset 2")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curves
plt.figure()
plt.plot(recall1, precision1, label="Dataset 1")
plt.plot(recall2, precision2, label="Dataset 2")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc="lower left")
plt.show()

# Plot TPR and FPR curves
plt.figure()
plt.plot(recall1, tpr1, label="Dataset 1 - TPR")
plt.plot(recall2, tpr2, label="Dataset 2 - TPR")
plt.plot(recall1, fpr1, label="Dataset 1 - FPR")
plt.plot(recall2, fpr2, label="Dataset 2 - FPR")
plt.xlabel("Recall")
plt.ylabel("Value")
plt.title("TPR and FPR Curves")
plt.legend(loc="lower right")
plt.show()

y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
y_scores = np.array([5.0, 1.0, 3.0, 2.0, 6.0, 4.0, 5.5, 6.2, 7.0, 5.3])
precision,recall, thresholds = precision_recall_curve(y_true, y_scores)
print(precision)
print(recall)

import matplotlib.pyplot as plt

# Dataset 1
fpr1 = [0.000, 0.000, 0.000, 0.000, 0.500, 0.500, 0.500, 0.500, 0.500]
tpr1 = [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
recall1 = [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
precision1 = [1.000, 1.000, 1.000, 1.000, 0.800, 0.833, 0.857, 0.875, 0.889]



# Plot ROC curves
plt.figure()
plt.plot(fpr1, tpr1, label="Dataset 1")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curves
plt.figure()
plt.plot(recall1, precision1, label="Dataset 1")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc="lower left")
plt.show()

# Plot TPR and FPR curves
plt.figure()
plt.plot(recall1, tpr1, label="Dataset 1 - TPR")
plt.plot(recall1, fpr1, label="Dataset 1 - FPR")
plt.xlabel("Recall")
plt.ylabel("Value")
plt.title("TPR and FPR Curves")
plt.legend(loc="lower right")
plt.show()