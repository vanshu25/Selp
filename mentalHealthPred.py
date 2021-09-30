import pandas as pd
df = pd.read_csv("Mental_Disorder_updated.data")

import pickle

import numpy as np
X = np.array(df.iloc[:, 0:10])
y = np.array(df.iloc[:, 10:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn import svm
import sklearn.model_selection as model_selection
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score


poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
poly_pred = poly.predict(X_test)

# poly_accuracy = accuracy_score(y_test, poly_pred)
# poly_f1 = f1_score(y_test, poly_pred, average='weighted')

# print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
# print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
pickle.dump(poly, open('MHP.pkl', 'wb'))