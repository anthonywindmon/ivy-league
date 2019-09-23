# Anthony Windmon - Support Vector Machine Classification
import csv
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics
#from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import scikitplot as skplt


#Loading dataset from desktop
dataset = 'C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_BREATH_4416_TEST4_MFCC_ONLY.csv'
raw_data = open(dataset, 'r')

#Reading shape of dataset
reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
list_name = list(reader)
data = numpy.array(list_name)
shape = data.shape
print(data.shape) #prints out number of samples, and number of features

#Creates dataframe for csv
df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_BREATH_4416_TEST4_MFCC_ONLY.csv')
print(df.head())
target = df['Class'] #uses the dataframe to single out the 'class' column as the target

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), target, test_size=0.4)
print('--------------------------TRAINING AND TESTING INFO-----------------------------------')
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

model = SVC(C=1.0, cache_size=200, degree=3,probability=True) #SVM Model
model.fit(X_train, y_train)
scores = model.score(X_test, y_test)
print('------------------------BASIC METRICS---------------------------')
print("Accuracy =", scores)

#Confusion Matrix
y_predicted = model.predict(X_test)
#print('y_predicted =', y_predicted)
confuse_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix: \n", confuse_matrix)

#10-Fold Cross Validation
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
print('----------------------------------------')
classifier_report = classification_report(y_test,y_predicted)
print(classifier_report)

print('------------------TRUE/FALSE POSITIVES/NEGATIVES----------------------')
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

#Paramter Tuning using cross_val_score
print('-------------------------PARAMETER TUNING-------------------------------')
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), target, test_size=0.4)
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

#SVM Test 2
#highest score (for BREATTH AND COUGH - TEST FOUR MFCC) cache_size=215, degree=5, random_state=4
model_two = SVC(C=1.0, cache_size=215, degree=5, random_state=4) #SVM Model
model_two.fit(X_train, y_train)
scores_two = model_two.score(X_test, y_test)
print('----------CLASSIFIER COMPARISON-----------')

print("Accuracy 1: =", scores)
print("Accuracy 2: =", scores_two)
print('-----')

#10-Fold Test 2
cv_scores = cross_val_score(model_two, df.drop(columns=['Class']), target, cv=10)
print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))
print('-----')

#K-Fold Test 2
#7 splits for highest scores (BREATH AND COUGH - TEST FOUR MFCC)
kf = KFold(n_splits=7, shuffle=True) #recommended number of splits is 5-10
for train, test in kf.split(df):
    #print('training = %s, testing = %s' %(train,test))
    train_data = numpy.array(df)[train]
    test_data = numpy.array(df)[test]
    kf_score_two = cross_val_score(model_two, df.drop(columns=['Class']), target, cv=kf)
print('Avg. K-Fold Scores (first) =', kf_score.mean())
print('Avg. K-Fold Scores (second) =', kf_score_two.mean())
print('-----')

#Stratified KFold Test 2
#6 splits for highest scores (BREATH AND COUGH - TEST FOUR MFCC)
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

#plot confusion matrix
fig = plt.figure()
csfont = {'fontname':'Times New Roman'}
plt.rcParams['font.family'] = 'Times New Roman'
sn.heatmap(confuse_matrix, fmt="d", annot=True, cbar=False,annot_kws={"size": 20})
ax = fig.add_subplot(111)
plt.title("Before & After Breath Confusion Matrix", fontsize=20,**csfont)
plt.xlabel('Predicted Label', fontsize=20,**csfont)
plt.ylabel('Truth Label', fontsize=20, **csfont)
ax.xaxis.set_ticklabels(['Before Breath', 'After Breath'], fontsize=20, horizontalalignment ='center')
ax.yaxis.set_ticklabels(['Before Breath', 'After Breath'], fontsize=20, verticalalignment='center')
plt.show()


#plt.figure()
#sn.heatmap(confuse_matrix, fmt="d", annot=True)
#plt.title("Before & After Breath Confusion Matrix")
#plt.set_xticklabels('Before', 'After')
#plt.xlabel('Predicted Label')
#plt.ylabel('Truth Label')
#plt.axis.YAxis.set_ticklabels('Before', 'After')
#plt.show()


#plot_decision_regions(X_train=X.values,
#                      y_train=y.values,
#                      clf=clf,
#                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
#plt.xlabel(X.columns[0], size=14)
#plt.ylabel(X.columns[1], size=14)
#plt.title('SVM Decision Region Boundary', size=16)
