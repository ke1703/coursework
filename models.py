from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

def kmeans(X_train, y_train, X_test, y_test):
    print('Обучение алгоритма k-Means...\n')
    clf = KMeans(random_state=0, max_iter=100, n_clusters=2)
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Отчет классификации для алгоритма k-Means: \n', classification_report(y_test, predictions))
    return clf

def gnb(X_train, y_train, X_test, y_test):
    print('Обучение алгоритма Gaussian NB...\n')
    clf = GaussianNB(var_smoothing=10)
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Отчет классификации для алгоритма Gaussian NB: \n', classification_report(y_test, predictions))
    return clf

def svc(X_train, y_train, X_test, y_test):
    print('Обучение алгоритма SVC...\n')
    clf = SVC(max_iter=10, gamma='auto', probability=True, kernel='linear')
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Отчет классификации для алгоритма SVC: \n', classification_report(y_test, predictions))
    return clf

def bagging(X_train, y_train, X_test, y_test, model):
    print('Обучение алгоритма Bagging...\n')
    clf = BaggingClassifier(base_estimator=model, n_estimators=10, random_state=0).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Отчет классификации для метода Bagging: \n', classification_report(y_test, predictions))
    return clf

def stacking(X_train, y_train, X_test, y_test, model1, model2):
    print('Обучение алгоритма Stacking...\n')
    estimators = [('kmeans', model1),('svc', model2)]
    clf = StackingClassifier(estimators=estimators)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Отчет классификации для метода Stacking: \n', classification_report(y_test, predictions))
    return clf

def adaboost(X_train, y_train, X_test, y_test):
    print('Обучение алгоритма Ada Boost...\n')
    clf = AdaBoostClassifier(algorithm='SAMME', n_estimators=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Отчет классификации для метода Ada Boost: \n', classification_report(y_test, predictions))
    return clf

def gradientboosting(X_train, y_train, X_test, y_test):
    print('Обучение алгоритма Gradient Boosting...\n')
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('Отчет классификации для метода Gradient Boosting: \n', classification_report(y_test, predictions))
    return clf