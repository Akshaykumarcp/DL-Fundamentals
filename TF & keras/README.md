#### Keras Model Life-Cycle
    Below is an overview of the 5 steps in the neural network model life-cycle in Keras:
    1. Define Network.
        - Keras sequentional models 
            - https://keras.io/api/models/sequential
        - keras functional models
            - https://keras.io/api/models/model/
    2. Compile Network.
    3. Fit Network.
    4. Evaluate Network.
    5. Make Predictions.

#### Common activation function for output layer
❼ Regression: Linear activation function, or linear (or None), and the number of neurons
matching the number of outputs.
❼ Binary Classification (2 class): Logistic activation function, or sigmoid, and one
neuron the output layer.
❼ Multiclass Classification (>2 class): Softmax activation function, or softmax, and
one output neuron per class value, assuming a one hot encoded output pattern.

#### standard loss functions for different predictive model types:
❼ Regression: Mean Squared Error or mean squared error.
❼ Binary Classification (2 class): Logarithmic Loss, also called cross-entropy or
binary crossentropy.
❼ Multiclass Classification (>2 class): Multiclass Logarithmic Loss or
categorical crossentropy.

#### commonly used optimization algorithms because of their generally better performance are:
❼ Stochastic Gradient Descent, or sgd, that requires the tuning of a learning rate and
momentum.
❼ Adam, or adam, that requires the tuning of learning rate.
❼ RMSprop, or rmsprop, that requires the tuning of learning rate.