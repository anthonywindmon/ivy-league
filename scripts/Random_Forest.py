# Anthony Windmon - Random Forest Classification

import csv
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.preprocessing import binarize
from sklearn.metrics import roc_auc_score


#Loading dataset from desktop
dataset = 'C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_COUGH_4416_CUT_TEST3_MFCC_ONLY.csv'
raw_data = open(dataset, 'r')

#Reading shape of dataset
reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
list_name = list(reader)
data = numpy.array(list_name)
shape = data.shape
print(data.shape) #prints out number of samples, and number of features

#Reading shape and target of dataset
df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_COUGH_4416_CUT_TEST3_MFCC_ONLY.csv')
print(df.head())
#shape = df.shape
target = df['Class'] #uses the dataframe to single out the 'class' column as the target

#These samples change every time we compile, therefore, the results will be different on each run
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), target, test_size=0.35,
                                                        random_state=42)
print('--------------------------TRAINING AND TESTING INFO-----------------------------------')
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

#RF Classification
model = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4) #RF Classifier
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print('------------------------BASIC METRICS---------------------------')
print("Accuracy =", result)

#Confusion Matrix
y_predicted = model.predict(X_test)
confuse_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix: \n", confuse_matrix)

# 10 fold cross validation
#clf = svm.SVC(kernel = 'linear', C=1)
cv_scores = cross_val_score(model, df.drop(columns=['Class']), target, cv=10)
print("10-Fold scores = ", cv_scores)
print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))

#K-Fold
kf = KFold(n_splits=6, shuffle=True) #recommended number of splits is 5-10
for train, test in kf.split(df):
    #print('training = %s, testing = %s' %(train,test))
    train_data = numpy.array(df)[train]
    test_data = numpy.array(df)[test]
    kf_score = cross_val_score(model, df.drop(columns=['Class']), target, cv=kf)
print('K-Fold Scores =', kf_score)
print('Avg. K-Fold Scores =', kf_score.mean())

#Stratified KFold
skf = StratifiedKFold(n_splits=7, shuffle=True)
for train, test in skf.split(df, target):
    #print('training = %s, testing = %s' %(train,test))
    skf_score = cross_val_score(model, df.drop(columns=['Class']), target, cv=skf)
print('Stratified K-Fold Scores =', skf_score)
print('Avg. Stratisfied K-Fold Scores =', skf_score.mean())

#precision, recall & f-measure
classifier_report = classification_report(y_test,y_predicted)
print(classifier_report)


print('------------------TRUE/FALSE POSITIVES/NEGATIVES----------------------')
#Which of these metrics should we focus on for our problem?
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

print('-----------------------------CHANGING THRESHOLD--------------------------------')
#Adjusting the classification threshold.
#The threshold is set at 0.5, however changing it can change the specificity and sensitivity of a model
#predicted_test = model.predict_proba(X_test)[0:10] #predict_proba is best used for classification models
#print('predicted values: \n', predicted_test)

#stores predicted probablities for class one
y_pred_prob = model.predict_proba(X_test)[:,1]
print('predicted values: \n', y_pred_prob)

#Changing threshold from 0.5, to 0.3
y_pred_class = binarize(y_pred_prob.reshape(1,-1), threshold=0.3)[0]
print('predicted values (after changing threshold): \n', y_pred_class)

new_threshold = y_pred_prob[0:10]
new_class_threshold = y_pred_class[0:10]
print('new class probablities: \n', new_threshold)
print('new class threshold: \n', new_class_threshold)
#confuse_matrix_two = confusion_matrix(y_test, y_pred_class)
#print('New CM: \n', confuse_matrix_two)

#LOOCV
#loocv = LeaveOneOut()
#for train, test in loocv.split(df):
    #print('training = %s, testing = %s' %(train,test))
    #loocv_score = cross_val_score(model, df.drop(columns=['Class']), target, cv=loocv)
#print('LOOCV =', loocv_score)

#Parameter Tuning
print('-------------------------PARAMETER TUNING-------------------------------')
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), target, test_size=0.35,
                                                    random_state=42)
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

#RF Classification Test 2
#highest scores (BREATH AND COUGH - TEST FOUR MFCC) - 150 trees, 15 max_depth (levels in each tree), max_features = None, min_samples_split =4, RS =0, min_samples_leaf=2
model_two = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4, max_features=None) #RF Classifier
model_two.fit(X_train, y_train)
result_two = model_two.score(X_test, y_test)

print('----------CLASSIFIER COMPARISON-----------')
print('Accuracy 1: ', result)
print('Accuracy 2: ', result_two)
print('-----')

#10-Fold Test 2
cv_scores = cross_val_score(model_two, df.drop(columns=['Class']), target, cv=10)
print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))
print('-----')

#K Fold Test 2
#5 splits for highest scores (BREATH AND COUGH - TEST FOUR MFCC)
kf = KFold(n_splits=6, shuffle=True) #recommended number of splits is 5-10
for train, test in kf.split(df):
    #print('training = %s, testing = %s' %(train,test))
    train_data = numpy.array(df)[train]
    test_data = numpy.array(df)[test]
    kf_score_two = cross_val_score(model_two, df.drop(columns=['Class']), target, cv=kf)
print('Avg. K-Fold Scores (first) =', kf_score.mean())
print('Avg. K-Fold Scores (second) =', kf_score_two.mean())
print('-----')

#Stratified KFold Test 2
#6 splits highest scores (BREATH AND COUGH - TEST FOUR MFCC)
skf = StratifiedKFold(n_splits=6, shuffle=True)
for train, test in skf.split(df, target):
    #print('training = %s, testing = %s' %(train,test))
    skf_score_two = cross_val_score(model_two, df.drop(columns=['Class']), target, cv=skf)
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

