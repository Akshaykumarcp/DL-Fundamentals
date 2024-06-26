# DL-Fundamentals
Deep Learning Fundamentals Awareness


### Activation functions

- Why do we need non-linear activation function?
  - Modeling Complex Relationships: Linear functions, by themselves, can only create straight lines. Real-world data, however, often has intricate and non-linear relationships. Non-linear activation functions allow neural networks to learn these complex patterns. For instance, imagine trying to model the relationship between image pixels and whether they represent a cat or a dog. A linear function wouldn't be able to capture the subtle non-linearities that distinguish these two classes.
  - Expressive Power: By stacking multiple layers with non-linear activations one after another, deep neural networks can learn increasingly complex functions. This is essential for tasks like image recognition and natural language processing, where the underlying data has many factors and features interacting in non-linear ways. Each non-linear layer acts like a building block, allowing the network to progressively learn more intricate structures in the data.
  - Gradient Flow: During training with backpropagation, linear activation functions can lead to a phenomenon called vanishing gradients. This happens because the gradients used to update the network weights become very small or even zero as they propagate backward through the layers. Non-linear activation functions help prevent this issue by introducing non-zero gradients throughout the network, allowing the backpropagation algorithm to efficiently update the weights and learn from the data.
- Example activation functions:
  - [Sigmoid](https://github.com/Akshaykumarcp/Neural-Network-from-scratch/blob/main/nn/numpy/sigmoid.ipynb)
  - [Softmax](https://github.com/Akshaykumarcp/Neural-Network-from-scratch/blob/main/nn/numpy/softmax.ipynb)
- Pytorch activation functions: [1](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity), [2](https://pytorch.org/docs/stable/nn.html#non-linear-activations-other).

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
- Pytorch [init](https://pytorch.org/docs/stable/nn.init.html)

### Debug NN
- [Grad checking](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%201/Gradient_Checking.ipynb)

### Underfit, Overfit, Good fit
- Underfit Fix:
  - Try Bigger network
  - Train Longer

- Overfit Fix:
  - More data
  - Regularization
    - [L2 Regularization](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%201/Regularization.ipynb)
    - [Dropout](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%201/Regularization.ipynb)
    - Other Regularization methods
      - Data augmentation
      - Early stopping

### Optimization problem

- Normalize inputs (converge smoothly)
- Vanishing/Exploding gradients
- Numerical approximation of gradients, gradient check
- Algorithms
  - Mini-batch gradient descent (GD)
    - Size = m; batch gradient descent
    - Size = 1; stochastic GD
    - Size = between 1 to m; example: 32; Mini batch
    - Choose mini batch size?
      - small dataset i,e less than 2000 - use batch GD
      - Mini-batch sizes: 64, 128, 256, 512, etc
      - Mini batch must fit within CPU/GPU memory
    - Exponentially weighted averages (EWA)
      - Vt = Beta 1 * Vt-1 + (1-beta 1) Thetat # Thetat is the previous temp
        - beta 1 = 0.9: appx 10 days temp
        - beta 1 = 0.98: appx 50 days
        - beta 1 = 0.5: appx 2 days
        - beta 1 = 0: standard GD
      - Bias correction
        - Vt / 1 - beta 1t
    - GD with momentum
      - Use EWA at the time of updating W and B
      - Hyperparams:
        - alpha
        - Beta 1 # usually 0.9
      - Note that:
        - The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps.
        - If $\beta = 0$, then this just becomes standard gradient descent without momentum.
      - How do you choose $\beta$?
        - The larger the momentum $\beta 1$ is, the smoother the update, because it takes the past gradients into account more. But if $\beta 1$ is too big, it could also smooth out the updates too much.
        - Common values for $\beta 1$ range from 0.8 to 0.999. If you don't feel inclined to tune this, $\beta 1 = 0.9$ is often a reasonable default.
        - Tuning the optimal $\beta 1$ for your model might require trying several values to see what works best in terms of reducing the value of the cost function $J$.
      - What you should remember:
        - Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
        - You have to tune a momentum hyperparameter $\beta$ and a learning rate $\alpha$.
    - RMSprop (Root Mean Square Propagation)
      - updated formula of EWA
        - beta 2 param introduced
      - updated formula for updating W and B
    - Adam (Adaptive moment estimation)
      - Combination of GD with momentum and RMSProp

      - **How does Adam work?**
        1. It calculates an exponentially weighted average of past gradients, and stores it in variables $v$ (before bias correction) and $v^{corrected}$ (with bias correction).
        2. It calculates an exponentially weighted average of the squares of the past gradients, and  stores it in variables $s$ (before bias correction) and $s^{corrected}$ (with bias correction).
        3. It updates parameters in a direction based on combining information from "1" and "2".

        The update rule is, for $l = 1, ..., L$:

        $$\begin{cases}
        v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\
        v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\
        s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\
        s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_2)^t} \\
        W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}
        \end{cases}$$
        where:
        - t counts the number of steps taken of Adam
        - L is the number of layers
        - $\beta_1$ and $\beta_2$ are hyperparameters that control the two exponentially weighted averages.
        - $\alpha$ is the learning rate
        - $\varepsilon$ is a very small number to avoid dividing by zero

      - Hyperparameters:
        - LR: needs to be tuned
        - Beta 1: 0.9
        - Beta 2: 0.999
        - Epsilon: 10 power -8
  - Learning rate decay
    - Manual
    - Exponential weight decay (formula based on epoch #)
      - if you set the decay to occur at every iteration, the learning rate goes to zero too quickly even if you start with a higher learning rate.
      - When you're training for a few epoch this doesn't cause a lot of troubles, but when the number of epochs is large the optimization algorithm will stop updating. One common fix to this issue is to decay the learning rate every few steps. This is called fixed interval scheduling.
    - Discrete staircase / Fixed interval scheduling
      - Calculate the new learning rate using exponential weight decay with fixed interval scheduling.
      - $$\alpha = \frac{1}{1 + decayRate \times \lfloor\frac{epochNum}{timeInterval}\rfloor} \alpha_{0}$$
  - Problem of local minima
    - Unlikely to get stuck in a bad local optima
    - Plateaus can make learning slow
  - Batch normalization (normalize activations)
  - Ref [implementation](https://github.com/Akshaykumarcp/coursera/blob/main/Deep%20Learning%20Specialization/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%20and%20Optimization/week%202/Optimization_methods.ipynb)
  - Pytorch [optimizers](https://pytorch.org/docs/stable/optim.html)


  ### Hyperparameters
  - List:
    - Learning rate, alpha
    - Number of interations
    - Number of hidden layers
    - Number of hidden units
    - Choice of activation function
    - Learning rate decay
    - Mini batch size
    - Momemtum beta 1 # 0.9
    - Adam, beta 1, beta 2 and epsilon # 0.9, 0.999, 10-8

  - How to sample values for params?
    - Choose random sampling and experiment
    - Coarse to fine

  - Appropriate scale to pick
    - uniform scale

  - Tune practices (pandas vs caviar)
    - One model (pandas)
      - Day wise observe cost function decrease
    - Train many models in parallel (caviar)
      - Plot model cost function

  - Batch normalization
    - Normalize actications in a network
    - for deeper models, train activations faster
    - Usually applied after computing Z
    - Why BN works?
