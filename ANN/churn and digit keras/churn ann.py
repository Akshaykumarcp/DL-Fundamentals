# Artificial Neural Network

#  Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
""" 
      RowNumber  CustomerId    Surname  CreditScore Geography  Gender  Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
0             1    15634602   Hargrave          619    France  Female   42       2       0.00              1          1               1        101348.88       1
1             2    15647311       Hill          608     Spain  Female   41       1   83807.86              1          0               1        112542.58       0
2             3    15619304       Onio          502    France  Female   42       8  159660.80              3          1               0        113931.57       1
3             4    15701354       Boni          699    France  Female   39       1       0.00              2          0               0         93826.63       0
4             5    15737888   Mitchell          850     Spain  Female   43       2  125510.82              1          1               1         79084.10       0
...         ...         ...        ...          ...       ...     ...  ...     ...        ...            ...        ...             ...              ...     ...
9995       9996    15606229   Obijiaku          771    France    Male   39       5       0.00              2          1               0         96270.64       0
9996       9997    15569892  Johnstone          516    France    Male   35      10   57369.61              1          1               1        101699.77       0
9997       9998    15584532        Liu          709    France  Female   36       7       0.00              1          0               1         42085.58       1
9998       9999    15682355  Sabbatini          772   Germany    Male   42       3   75075.31              2          1               0         92888.52       1
9999      10000    15628319     Walker          792    France  Female   28       4  130142.79              1          1               0         38190.78       0

[10000 rows x 14 columns] """

X_df=dataset.iloc[:, 3:13]
""" 
     CreditScore Geography  Gender  Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary
0             619    France  Female   42       2       0.00              1          1               1        101348.88
1             608     Spain  Female   41       1   83807.86              1          0               1        112542.58
2             502    France  Female   42       8  159660.80              3          1               0        113931.57
3             699    France  Female   39       1       0.00              2          0               0         93826.63
4             850     Spain  Female   43       2  125510.82              1          1               1         79084.10
...           ...       ...     ...  ...     ...        ...            ...        ...             ...              ...
9995          771    France    Male   39       5       0.00              2          1               0         96270.64
9996          516    France    Male   35      10   57369.61              1          1               1        101699.77
9997          709    France  Female   36       7       0.00              1          0               1         42085.58
9998          772   Germany    Male   42       3   75075.31              2          1               0         92888.52
9999          792    France  Female   28       4  130142.79              1          1               0         38190.78 """
#X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13]
""" 
0       1
1       0
2       1
3       0
4       0
       ..
9995    0
9996    0
9997    1
9998    1
9999    0
Name: Exited, Length: 10000, dtype: int64 """

 #Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
X=pd.get_dummies(X_df)
""" 
      CreditScore  Age  Tenure    Balance  NumOfProducts  HasCrCard  ...  EstimatedSalary  Geography_France  Geography_Germany  Geography_Spain  Gender_Female  Gender_Male
0             619   42       2       0.00              1          1  ...        101348.88                 1                  0                0              1            0    
1             608   41       1   83807.86              1          0  ...        112542.58                 0                  0                1              1            0    
2             502   42       8  159660.80              3          1  ...        113931.57                 1                  0                0              1            0    
3             699   39       1       0.00              2          0  ...         93826.63                 1                  0                0              1            0    
4             850   43       2  125510.82              1          1  ...         79084.10                 0                  0                1              1            0    
...           ...  ...     ...        ...            ...        ...  ...              ...               ...                ...              ...            ...          ...    
9995          771   39       5       0.00              2          1  ...         96270.64                 1                  0                0              0            1    
9996          516   35      10   57369.61              1          1  ...        101699.77                 1                  0                0              0            1    
9997          709   36       7       0.00              1          0  ...         42085.58                 1                  0                0              1            0    
9998          772   42       3   75075.31              2          1  ...         92888.52                 0                  1                0              0            1    
9999          792   28       4  130142.79              1          1  ...         38190.78                 1                  0                0              1            0    

[10000 rows x 13 columns] """

X=X.drop(['Geography_France','Gender_Female'], axis=1)
""" 
array([[619.,  42.,   2., ...,   0.,   0.,   0.],
       [608.,  41.,   1., ...,   0.,   1.,   0.],
       [502.,  42.,   8., ...,   0.,   0.,   0.],
       ...,
       [709.,  36.,   7., ...,   0.,   0.,   0.],
       [772.,  42.,   3., ...,   1.,   0.,   1.],
       [792.,  28.,   4., ...,   0.,   0.,   0.]]) """

X1=X
""" 
      CreditScore  Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Geography_Germany  Geography_Spain  Gender_Male
0             619   42       2       0.00              1          1               1        101348.88                  0                0            0
1             608   41       1   83807.86              1          0               1        112542.58                  0                1            0
2             502   42       8  159660.80              3          1               0        113931.57                  0                0            0
3             699   39       1       0.00              2          0               0         93826.63                  0                0            0
4             850   43       2  125510.82              1          1               1         79084.10                  0                1            0
...           ...  ...     ...        ...            ...        ...             ...              ...                ...              ...          ...
9995          771   39       5       0.00              2          1               0         96270.64                  0                0            1
9996          516   35      10   57369.61              1          1               1        101699.77                  0                0            1
9997          709   36       7       0.00              1          0               1         42085.58                  0                0            0
9998          772   42       3   75075.31              2          1               0         92888.52                  1                0            1
9999          792   28       4  130142.79              1          1               0         38190.78                  0                0            0

[10000 rows x 11 columns] """

X=X.values
""" 
array([[619.,  42.,   2., ...,   0.,   0.,   0.],
       [608.,  41.,   1., ...,   0.,   1.,   0.],
       [502.,  42.,   8., ...,   0.,   0.,   0.],
       ...,
       [709.,  36.,   7., ...,   0.,   0.,   0.],
       [772.,  42.,   3., ...,   1.,   0.,   1.],
       [792.,  28.,   4., ...,   0.,   0.,   0.]]) """

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
""" 
array([[ 0.16958176, -0.46460796,  0.00666099, ..., -0.5698444 ,
         1.74309049, -1.09168714],
       [-2.30455945,  0.30102557, -1.37744033, ...,  1.75486502,
        -0.57369368,  0.91601335],
       [-1.19119591, -0.94312892, -1.031415  , ..., -0.5698444 ,
        -0.57369368, -1.09168714],
       ...,
       [ 0.9015152 , -0.36890377,  0.00666099, ..., -0.5698444 ,
        -0.57369368,  0.91601335],
       [-0.62420521, -0.08179119,  1.39076231, ..., -0.5698444 ,
         1.74309049, -1.09168714],
       [-0.28401079,  0.87525072, -1.37744033, ...,  1.75486502,
        -0.57369368, -1.09168714]]) """

X_test = sc.transform(X_test)
""" 
array([[-0.55204276, -0.36890377,  1.04473698, ...,  1.75486502,
        -0.57369368, -1.09168714],
       [-1.31490297,  0.10961719, -1.031415  , ..., -0.5698444 ,
        -0.57369368, -1.09168714],
       [ 0.57162971,  0.30102557,  1.04473698, ..., -0.5698444 ,
         1.74309049, -1.09168714],
       ...,
       [-0.74791227, -0.27319958, -1.37744033, ..., -0.5698444 ,
         1.74309049,  0.91601335],
       [-0.00566991, -0.46460796, -0.33936434, ...,  1.75486502,
        -0.57369368,  0.91601335],
       [-0.79945688, -0.84742473,  1.04473698, ...,  1.75486502,
        -0.57369368,  0.91601335]]) """

# Now let's make the ANN!

# Importing the Keras libraries and packages
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', 
                     activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)
""" 
Epoch 1/20
800/800 [==============================] - 0s 460us/step - loss: 0.4551 - accuracy: 0.7972
Epoch 2/20
800/800 [==============================] - 0s 482us/step - loss: 0.4161 - accuracy: 0.8278
Epoch 3/20
800/800 [==============================] - 0s 467us/step - loss: 0.4083 - accuracy: 0.8313
Epoch 4/20
800/800 [==============================] - 0s 452us/step - loss: 0.4024 - accuracy: 0.8349
Epoch 5/20
800/800 [==============================] - 0s 460us/step - loss: 0.3862 - accuracy: 0.8395
Epoch 6/20
800/800 [==============================] - 0s 453us/step - loss: 0.3678 - accuracy: 0.8485
Epoch 7/20
800/800 [==============================] - 0s 458us/step - loss: 0.3579 - accuracy: 0.8530
Epoch 8/20
800/800 [==============================] - 0s 454us/step - loss: 0.3519 - accuracy: 0.8559
Epoch 9/20
800/800 [==============================] - 0s 449us/step - loss: 0.3449 - accuracy: 0.8572
Epoch 10/20
800/800 [==============================] - 0s 457us/step - loss: 0.3413 - accuracy: 0.8595
Epoch 11/20
800/800 [==============================] - 0s 451us/step - loss: 0.3394 - accuracy: 0.8625
Epoch 12/20
800/800 [==============================] - 0s 451us/step - loss: 0.3382 - accuracy: 0.8584
Epoch 13/20
800/800 [==============================] - 0s 446us/step - loss: 0.3358 - accuracy: 0.8641
Epoch 14/20
800/800 [==============================] - 0s 449us/step - loss: 0.3349 - accuracy: 0.8645
Epoch 15/20
800/800 [==============================] - 0s 451us/step - loss: 0.3343 - accuracy: 0.8646
Epoch 16/20
800/800 [==============================] - 0s 469us/step - loss: 0.3342 - accuracy: 0.8625
Epoch 17/20
800/800 [==============================] - 0s 452us/step - loss: 0.3320 - accuracy: 0.8640
Epoch 18/20
800/800 [==============================] - 0s 457us/step - loss: 0.3319 - accuracy: 0.8626
Epoch 19/20
800/800 [==============================] - 0s 456us/step - loss: 0.3312 - accuracy: 0.8649
Epoch 20/20
800/800 [==============================] - 0s 452us/step - loss: 0.3293 - accuracy: 0.8619
<tensorflow.python.keras.callbacks.History object at 0x0000026D8EF69308> """
#  Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
""" 
array([[0.23277977],
       [0.30433112],
       [0.11409187],
       ...,
       [0.21590808],
       [0.16846022],
       [0.23988283]], dtype=float32) """

y_pred = (y_pred > 0.5)
""" 
array([[False],
       [False],
       [False],
       ...,
       [False],
       [False],
       [False]]) """

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred) # 0.865

classifier.save('churn_model.h5')