# Anthony Windmon - Logistic Regression Model

import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sn

#Load dataset
dataset = 'C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_COUGH_4416_CUT_TEST3_MFCC_ONLY.csv'
raw_data = open(dataset, 'r')

#Reading shape of dataset
reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
list_name = list(reader)
data = numpy.array(list_name)
shape = data.shape
print(data.shape) #prints out number of samples, and number of features

#Creates dataframe for csv
df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_COUGH_4416_CUT_TEST3_MFCC_ONLY.csv')
print(df.head())
target = df['Class'] #uses the dataframe to single out the 'class' column as the target

#changes target to binary values. This allows me to calculate the AUC & ROC later on
binary_labels = []
for class_label in target:
    new_class_label = 0 if class_label == 'BEFORE_COUGH' else 1
    binary_labels.append(new_class_label)
print(binary_labels)

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), binary_labels, test_size=0.20)
print('--------------------------TRAINING AND TESTING INFO-----------------------------------')
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample


model = LogisticRegression(max_iter=150, class_weight='balanced')
model.fit(X_train, y_train) #training the model_selection
model.predict(X_test)
scores = model.score(X_test,y_test)
print('------------------------BASIC METRICS---------------------------')
print("Accuracy = ", scores)

#Confusion Matrix
y_predicted = model.predict(X_test)
confuse_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix: \n", confuse_matrix)

print('----------TEST-----------')
#scores_three = model.score(y_test, y_predicted)
#print("Accuracy (using different parameters) =", scores_three)

#probability that class will be predicted as before or after cough/breath
#prob_of_data = model.predict_proba(X_test)
#print("Probability of data: \n", prob_of_data)

cv_scores = cross_val_score(model, df.drop(columns=['Class']), binary_labels, cv=10)
print("10-Fold scores = ", cv_scores)
print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))


#K-Fold
kf = KFold(n_splits=6, shuffle=True) #recommended number of splits is 5-10
for train, test in kf.split(df):
    #print('training = %s, testing = %s' %(train,test))
    train_data = numpy.array(df)[train]
    test_data = numpy.array(df)[test]
    kf_score = cross_val_score(model, df.drop(columns=['Class']), binary_labels, cv=kf)
print('K-Fold Scores =', kf_score)
print('Avg. K-Fold Scores =', kf_score.mean())

#Stratified KFold
skf = StratifiedKFold(n_splits=7, shuffle=True)
for train, test in skf.split(df, target):
    #print('training = %s, testing = %s' %(train,test))
    skf_score = cross_val_score(model, df.drop(columns=['Class']), binary_labels, cv=skf)
print('Stratified K-Fold Scores =', skf_score)
print('Avg. Stratisfied K-Fold Scores =', skf_score.mean())

#precision, recall & f-measure
print('----------------------------------------')
print(y_test)
print(y_predicted)
classifier_report = classification_report((y_test), (y_predicted))
print(classifier_report)
print('----------------------------------------')

#true & false negatives and positives
true_neg = confuse_matrix[0][0]
print('true negatives =', true_neg)
false_neg = confuse_matrix[1][0]
print('false negatives =', false_neg)
true_pos = confuse_matrix[1][1]
print('true positives =', true_pos)
false_pos = confuse_matrix[0][1]
print('false positives =', false_pos)

#sensitivity, recall, TP rate
sensitivity = true_pos/ float(true_pos+false_neg)
print('sensitivity =', sensitivity)

#specificity, TN rate
specificity = true_neg/ float(true_neg+false_pos)
print('specificity =', specificity)

#FP Rate
fp_rate = false_pos/ float(false_pos+true_neg)
print('false postive rate =', fp_rate)

#FN Rate
fn_rate = false_neg/ float(false_neg+true_pos)
print('false negative rate =', fn_rate)

print('-------------------------PARAMETER TUNING-------------------------------')
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), binary_labels, test_size=0.35,
                                                    random_state=42)
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

model_two = LogisticRegression(max_iter=150, class_weight='balanced')
model_two.fit(X_train, y_train)
scores_two = model_two.score(X_test, y_test)

print('----------CLASSIFIER COMPARISON-----------')
print('Accuracy 1: ', scores)
print('Accuracy 2: ', scores_two)
print('-----')

#10-Fold Test 2
cv_scores = cross_val_score(model_two, df.drop(columns=['Class']), binary_labels, cv=10)
print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))
print('-----')

kf = KFold(n_splits=6, shuffle=True) #recommended number of splits is 5-10
for train, test in kf.split(df):
    #print('training = %s, testing = %s' %(train,test))
    train_data = numpy.array(df)[train]
    test_data = numpy.array(df)[test]
    kf_score_two = cross_val_score(model_two, df.drop(columns=['Class']), binary_labels, cv=kf)
print('Avg. K-Fold Scores (first) =', kf_score.mean())
print('Avg. K-Fold Scores (second) =', kf_score_two.mean())
print('-----')

#Stratified KFold Test 2
skf = StratifiedKFold(n_splits=6, shuffle=True)
for train, test in skf.split(df, target):
    #print('training = %s, testing = %s' %(train,test))
    skf_score_two = cross_val_score(model_two, df.drop(columns=['Class']), binary_labels, cv=skf)
print('Avg. Stratisfied K-Fold Scores (first) =', skf_score.mean())
print('Avg. Stratisfied K-Fold Scores (second) =', skf_score_two.mean())

#Confusion Matrix
y_predicted_two = model_two.predict(X_test)
confuse_matrix_two = confusion_matrix(y_test, y_predicted_two)
print("Confusion Matrix: \n", confuse_matrix_two)

#precision, recall & f-measure
classifier_report = classification_report(y_test,y_predicted_two)
print(classifier_report)

true_neg = confuse_matrix_two[0][0]
print('true negatives =', true_neg)
false_neg = confuse_matrix_two[1][0]
print('false negatives =', false_neg)
true_pos = confuse_matrix_two[1][1]
print('true positives =', true_pos)
false_pos = confuse_matrix_two[0][1]
print('false positives =', false_pos)

#sensitivity, recall, TP rate
sensitivity = true_pos/ float(true_pos+false_neg)
print('sensitivity =', sensitivity)

#specificity, TN rate
specificity = true_neg/ float(true_neg+false_pos)
print('specificity =', specificity)

#FP Rate
fp_rate = false_pos/ float(false_pos+true_neg)
print('false postive rate =', fp_rate)

#FN Rate
fn_rate = false_neg/ float(false_neg+true_pos)
print('false negative rate =', fn_rate)

print('------------AUC/ROC--------------')
y_pred_proba = model.predict_proba(X_test)[::,1]
#print('y_pred_proba =', y_pred_proba)
#print('y_test =', y_test)

#plot confusion matrix for cough
fig = plt.figure()
csfont = {'fontname':'Times New Roman'}
plt.rcParams['font.family'] = 'Times New Roman'
sn.heatmap(confuse_matrix, fmt="d", annot=True, cbar=False,annot_kws={"size": 20})
ax = fig.add_subplot(111)
plt.title("Before & After Cough Confusion Matrix", fontsize=20,**csfont)
plt.xlabel('Predicted Label', fontsize=20,**csfont)
plt.ylabel('Truth Label', fontsize=20, **csfont)
ax.xaxis.set_ticklabels(['Before Cough', 'After Cough'], fontsize=20, horizontalalignment ='center')
ax.yaxis.set_ticklabels(['Before Cough', 'After Cough'], fontsize=20, verticalalignment='center')
plt.show()

#AUC & ROC
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print('AUC =', auc)
plt.plot(fpr,tpr,label='auc='+str(auc))
lw = 2
plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
plt.title('ROC of Before & After Breath Using Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
