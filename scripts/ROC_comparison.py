#Anthony Windmon -- ROC Curve comparison
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib as mpl

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#Reading shape and target of dataset
#df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_COUGH_4416_CUT_TEST3_MFCC_ONLY.csv')
df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_BREATH_4416_TEST4_MFCC_ONLY.csv')
print(df.head())
#shape = df.shape
target = df['Class'] #uses the dataframe to single out the 'class' column as the target

feature_names = df.columns.tolist()
#feature_names = df.drop(columns=['Class'])
feature_names.remove('Class')
#print('feature names =', feature_names)

#changes target to binary values
binary_labels = []
for class_label in target:
    new_class_label = 0 if class_label == 'BEFORE_BREATH' else 1
    #new_class_label = 0 if class_label == 'BEFORE_COUGH' else 1
    binary_labels.append(new_class_label)
#print(binary_labels)

#These samples change every time we compile, therefore, the results will be different on each run
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), binary_labels, test_size=0.35,
                                                        random_state=42)

#RF Classification
model_rf = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4) #RF Classifier
model_rf.fit(X_train, y_train)
y_pred_prob_rf = model_rf.predict_proba(X_test)[:,1]

#kNN classification
model_knn = KNeighborsClassifier(n_neighbors = 3)
model_knn.fit(X_train, y_train)
y_pred_prob_knn = model_knn.predict_proba(X_test)[:,1]

#SVM classification
model_svm = SVC(C=1.0, cache_size=200, degree=3,probability=True, gamma='auto') #SVM Model
model_svm.fit(X_train, y_train)
y_pred_prob_svm = model_svm.predict_proba(X_test)[:,1]

#Logistic Regression classification
model_reg = LogisticRegression(max_iter=150, class_weight='balanced',solver='lbfgs')
model_reg.fit(X_train, y_train) #training the model_selection
y_pred_prob_reg = model_reg.predict_proba(X_test)[:,1]

#MLP classification
model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), activation = 'relu',
                        learning_rate = 'constant', learning_rate_init = 0.001, max_iter=200, momentum=0.9,
                            batch_size = 'auto', beta_1=0.9, beta_2=0.99, early_stopping=False,random_state=1,
                                power_t=0.5, validation_fraction=0.1)
model_mlp.fit(X_train, y_train)
y_pred_prob_mlp = model_mlp.predict_proba(X_test)[:,1]

def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    roc_auc = roc_auc_score(y_test, pred_proba)
    return fpr, tpr, roc_auc

fpr, tpr, roc_auc = roc_curve_and_score(y_test, y_pred_prob_svm)
plt.plot(fpr, tpr, color='red', lw=2,
         label='Support Vector Machine ROC AUC={0:.3f}'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, y_pred_prob_knn)
plt.plot(fpr, tpr, color='green', lw=2,
         label='k-Nearest Neighbor ROC AUC={0:.3f}'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, y_pred_prob_rf)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='Random Forest ROC AUC={0:.3f}'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, y_pred_prob_reg)
plt.plot(fpr, tpr, color='blue', lw=2,
         label='Logistic Regression ROC AUC={0:.3f}'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, y_pred_prob_mlp)
plt.plot(fpr, tpr, color='darkviolet', lw=2,
         label='Multilayer Preceptron ROC AUC={0:.3f}'.format(roc_auc))

plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
mpl.rc('font', family = 'Times New Roman')
plt.legend(loc="lower right",fontsize=13)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
csfont = {'fontname':'Times New Roman'}
plt.title('Before & After Breath ROC Curve Comparision',**csfont,fontsize=20)
plt.xlabel('False Positive Rate',**csfont,fontsize=20)
plt.ylabel('True Positive Rate',**csfont,fontsize=20)
plt.show()
