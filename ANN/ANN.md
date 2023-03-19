#### Training ANN with Stochastic GD:
- Randomly initialise the weights to small numbers close to 0.
- Input the first observation of your dataset in input layer, each feature in one input node.
- Forward-prop from left to right, the neuron are activated in a way that the impact of each neuron's activation is limited by the weights. Propogate the activations until getting the predicted result y.
- Compare the predicted result to actual result. Measure the generated error.
- Back-prop from right to left, the error is back-propogated. Update the weights according to how much they are responsible for the error. The LR decides by how much we update the weights.
- Repeat Step 1 to 5 and update weights after each (batch in batch learning) observation.
- When the whole training set passed through the ANN, that makes an epoch. Redo more epochs.

### Sequential Model Programming Flow in Keras
- Create a Sequential model
- Add Hidden Layers via the .add() method
- Add Output Layer via the .add() method
- Compile the Network (Deciding Optimizer & Loss Function
- Train the Network
- Test the Network
- Performance Metrics
