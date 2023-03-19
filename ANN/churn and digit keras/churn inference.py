import numpy as np
from tensorflow.keras.models import load_model
churn_model = load_model('churn_model.h5')

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Credit Score: 600
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
Geography: France
Gender: Male"""
new_prediction = churn_model.predict(sc.fit_transform
                                     (np.array([[600.0, 40,3, 60000, 2, 1, 1, 50000, 0, 0, 1]])))
# array([[0.15256485]], dtype=float32)
new_prediction = (new_prediction > 0.5)
# array([[False]])
#new_prediction2 = churn_model.predict(sc.fit_transform
#                                     (np.array([[502.0,42,8, 159660, 3, 1, 0, 113931, 0, 0, 0]])))
#new_prediction1 = (new_prediction2 > 0.5)