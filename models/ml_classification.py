from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# KNeighborsClassifier
def KNeighbors(x_train_all, x_test, y_train_all, y_test):
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(x_train_all, y_train_all)
    KNN_pred = KNN.predict(x_test)
    print("1.KNeighborsClassifier")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, KNN_pred))
    print(metrics.classification_report(y_test, KNN_pred))
    return KNN_pred


# BaggingClassifier
def Bagging(x_train_all, x_test, y_train_all, y_test):
    BAG = BaggingClassifier(random_state=222, n_estimators=92)
    BAG.fit(x_train_all, y_train_all)
    BAG_pred = BAG.predict(x_test)
    print("2.BaggingClassifier")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, BAG_pred))
    print(metrics.classification_report(y_test, BAG_pred))
    return BAG_pred


# DecisionTreeClassifier
def DecisionTree(x_train_all, x_test, y_train_all, y_test):
    DT = DecisionTreeClassifier(random_state=12)
    DT.fit(x_train_all, y_train_all)
    DT_pred = DT.predict(x_test)
    print("3.DecisionTreeClassifier")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, DT_pred))
    print(metrics.classification_report(y_test, DT_pred))
    return DT_pred


# RandomForestClassifier
def RandomForest(x_train_all, x_test, y_train_all, y_test):
    RFC = RandomForestClassifier(n_estimators=666, random_state=82)
    RFC.fit(x_train_all, y_train_all)
    RFC_pred = RFC.predict(x_test)
    print("4.RandomForestClassifier")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, RFC_pred))
    print(metrics.classification_report(y_test, RFC_pred))
    return RFC_pred


# svm
def svmClassification(x_train_all, x_test, y_train_all, y_test):
    linearsvc = SVC(kernel='linear')
    model = linearsvc.fit(x_train_all, y_train_all)
    liner_svm_pred = model.predict(x_test)
    print("5.svm")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, liner_svm_pred))
    print(metrics.classification_report(y_test, liner_svm_pred))
    return liner_svm_pred


# 贝叶斯
def Gaussian(x_train_all, x_test, y_train_all, y_test):
    gnb = GaussianNB()
    model = gnb.fit(x_train_all, y_train_all)
    gnb_pred = model.predict(x_test)
    print("6.GaussianNB高斯朴素贝叶斯")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, gnb_pred))
    print(metrics.classification_report(y_test, gnb_pred))
    return gnb_pred


# 逻辑回归
def LogisticRegress(x_train_all, x_test, y_train_all, y_test):
    lr = LogisticRegression()
    model = lr.fit(x_train_all, y_train_all)
    lr_pred = model.predict(x_test)
    print("7.逻辑回归")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, lr_pred))
    print(metrics.classification_report(y_test, lr_pred))
    return lr_pred


# GradientBoostingClassifier
def GradientBoosting(x_train_all, x_test, y_train_all, y_test):
    GradBost = GradientBoostingClassifier(random_state=15)
    GradBost.fit(x_train_all, y_train_all)
    GradBost_pred = GradBost.predict(x_test)
    print("8.GradientBoostingClassifier")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, GradBost_pred))
    print(metrics.classification_report(y_test, GradBost_pred))
    return GradBost_pred


# AdaBoostClassifier
def AdaBoost(x_train_all, x_test, y_train_all, y_test):
    ADA = AdaBoostClassifier(random_state=37)
    ADA.fit(x_train_all, y_train_all)
    ADA_pred = ADA.predict(x_test)
    print("9.AdaBoostClassifier")
    print('ACC: %.4f' % metrics.accuracy_score(y_test, ADA_pred))
    print(metrics.classification_report(y_test, ADA_pred))
    return ADA_pred