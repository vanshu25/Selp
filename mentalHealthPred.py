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
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
knn = KNeighborsClassifier(n_neighbors=12)
sv = knn.fit(X, y)

pickle.dump(sv, open('MHP.pkl', 'wb'))