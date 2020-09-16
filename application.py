import models
import os
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

def RunApp():
    namedata = 'data.csv'
    scaler = StandardScaler()
    data = pd.read_csv(f'Data/{namedata}')
    y = data['is_canceled'].astype('int')
    X = data.drop(['is_canceled'], axis=1)
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    kmeans = models.kmeans(X_train, y_train, X_test, y_test)
    gnb = models.gnb(X_train, y_train, X_test, y_test)
    svc = models.svc(X_train, y_train, X_test, y_test)
    bagging = models.bagging(X_train, y_train, X_test, y_test, svc)
    stacking = models.stacking(X_train, y_train, X_test, y_test, gnb, svc)
    adaboost = models.adaboost(X_train, y_train, X_test, y_test)
    gradientboosting = models.gradientboosting(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    isNotRun = True
    while isNotRun:
        print('Что Вы хотите сделать? (Введите цифру)\n1. Сравнить ансамблевые методы\n2. Выйти из приложения\n')
        action = input('->')
        if action == '1':
            os.system('cls')
            RunApp()
        elif action == '2':
            break
        else:
            os.system('cls')
            print('Некорректный ввод! Попробуйте ещё раз.\n')