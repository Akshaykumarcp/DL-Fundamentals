# DL-Fundamentals
All about Deep Learning Fundamentals


### Activation functions

- Why do we need non-linear activation function?
  - Modeling Complex Relationships: Linear functions, by themselves, can only create straight lines. Real-world data, however, often has intricate and non-linear relationships. Non-linear activation functions allow neural networks to learn these complex patterns. For instance, imagine trying to model the relationship between image pixels and whether they represent a cat or a dog. A linear function wouldn't be able to capture the subtle non-linearities that distinguish these two classes.
  - Expressive Power: By stacking multiple layers with non-linear activations one after another, deep neural networks can learn increasingly complex functions. This is essential for tasks like image recognition and natural language processing, where the underlying data has many factors and features interacting in non-linear ways. Each non-linear layer acts like a building block, allowing the network to progressively learn more intricate structures in the data.
  - Gradient Flow: During training with backpropagation, linear activation functions can lead to a phenomenon called vanishing gradients. This happens because the gradients used to update the network weights become very small or even zero as they propagate backward through the layers. Non-linear activation functions help prevent this issue by introducing non-zero gradients throughout the network, allowing the backpropagation algorithm to efficiently update the weights and learn from the data.
- Example activation functions:
  - [Sigmoid](https://github.com/Akshaykumarcp/Neural-Network-from-scratch/blob/main/nn/numpy/sigmoid.ipynb)
  - [Softmax](https://github.com/Akshaykumarcp/Neural-Network-from-scratch/blob/main/nn/numpy/softmax.ipynb) 

### Weight Initialization
- What happens if you initialize weights to zero?
  - Symmetry Problem: With zero weights, neurons in the same layer will receive identical inputs (assuming the same activation for previous layers). This leads to them making the same calculations and outputs, essentially turning them into redundant copies of each other. The network effectively behaves like a single neuron model, limiting its ability to learn complex patterns.
  - Vanishing Gradients: Backpropagation relies on gradients to update weights during training. With zero initial weights, the initial gradients will also be zero. This can lead to the "vanishing gradients" problem, where gradients become very small or zero as they propagate backward through the network. This effectively halts learning in the earlier layers as they receive no updates to their weights.
  - Here's a breakdown of the consequences:
    - Stuck in Local Minima: Due to the lack of diverse weight initialization, the network might get stuck in a suboptimal point on the error surface (local minima) instead of finding the global minimum that represents the best fit for the data.
    - Slow or No Learning: Vanishing gradients prevent the network from effectively learning from the training data, hindering its ability to improve performance.
- What happens if you initialize weights to large #?
  - Saturation of Activation Functions: Activation functions such as sigmoid or tanh saturate at the extremes (i.e., approach 0 or 1), meaning that large inputs can result in outputs that are very close to these extreme values. This saturation can hinder the network's ability to learn effectively, as it reduces the gradient flow and slows down learning. 
    - when W is large, Z is larger.
    - if using sigmoid/tanh, we end up in flat part of the curve. therefore slope/gradient is very small and end up in slow learning
  - Exploding Gradients: This problem arises because large initial weights can cause gradients to become very large during backpropagation. This can happen especially with activation functions that have steep slopes in certain regions (like ReLU). As gradients are multiplied through the layers while backpropagating, large initial weights can lead to exponentially increasing gradients in the earlier layers.
  - Numerical Instability: Extremely large gradients can cause numerical instability during training, leading to issues like inaccurate weight updates and model divergence (failing to converge on a good solution).
  - Poor Generalization: Networks with excessively large weights are prone to overfitting, as they may memorize the training data instead of learning generalizable patterns. This can result in poor performance on unseen data, as the network fails to generalize beyond the training set.
- Zero, random and He weight initializations [implementation](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%201/Initialization.ipynb)

### Debug NN
- [Grad checking](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%201/Gradient_Checking.ipynb)

### Underfit, Overfit, Good git 
- Overfit Fix:
  - Regularization
    - [L2 Regularization](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%201/Regularization.ipynb)
    - [Dropout](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%201/Regularization.ipynb)
