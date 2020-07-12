#### Short note on deep learning introduction.
* When we try to solve a problem using a network of Perceptrons, then we are using deep learning approach.
* In Neural networks(NN), we have two parts. Forward Propagation and Backward Propagation.
#### Forward Propagation
* We have a network of input layer, hidden layers and output layer.
* Number of Perceptrons in input and output is equal to number of inputs and no of outputs respectively.
* We can add bias to input and to hidden layers also.
* We propagate our data in forward pass and assign some initital values for weights and bias which are learned in backward pass.
* <b>Weighted sum of inputs (xw+b) is passed to each perceptron and activation function is applied to generate output.</b>
* While passing data from input to hidden layer, each perceptron receive multiple permutation and combination of inputs, so that it can learn complex data also.
* Once data is propagated and output is generated, <b>error = target - predicted</b> is calculated.
* Weights are updated in backward propagation to reduce this error using optimizers.
* Optimizer is an algorithm which uses <b>derivatives of cost function and activation function to update weights and minimize error</b>
* one pass of forward and backward propagation is known as one <b>epoch</b>.
#### Backward Propagation
* Error is there because we have given some weights and biases but these values are not correct as per our dataset. So, network will try to change each weight and bias.
* <b>Chain rule Concept used in backpropagation: Error is differentiable with respect to output of Perceptron. Output of any Perceptron is based on it's input(wx+b). Input is differentiable w.r.t to weight.</b>
* For output layer, cost function and activation function is directly differentiable because target is known.
* For weight updation in Hidden layers, again multiple chain rule is applied which takes input from the previous layer and updated weights from the next layer.
* <b>Suppose I am trying to update weight in m<sup>th</sup> layer. Then it's differential calculation will depend on input from m-1<sup>th</sup> layer and output from m+1<sup>th</sup> layer.</b>
* Once we update all the learning parameters, data is agin forward propagated, error is calculaetd and weights are updated.
* This process continues, untill learning stops means weights are not updating and remain same or error is not reducing further.
#### Importance of Bias
* Bias is a learning parameter which is applied per Perceptron genrerally but we can also take one bias per layer. 
* Purpose of bias is to provide more flexibility to network to learn.
* <b>Bias is used to shift activation function in space similar to intercept is used to shift line across Y-axis in Linear regression.</b> 
#### Importance of Activation Function
* It normalize the data in forward pass. Sigmoid and Tanh functions limit the data between certain ranges and so normalize the data.
* Differentiation of activation function is used in backpropagation to update weights. That is why <b>Activation function should be differentiable</b> is a requirement.
* If we do not add activation function then It will be a linear operation(wx+b). It is activation function which help the netwrok to learn complex patterns.
#### Why Vanishing Gradient Problem occur?
* Whenever we differentiate a function, we are basically reducing the degree of function by 1 and at last it become zero. So, In chain rule we keep on differentating the activation function and multiply it with other values.
* Second in Sigmoid and Tanh activation functions, derivative of sigmoid and Tanh is already very small. For sigmoid maximum derivate value is 0.25. 
* Third, Sigmoid function always gives an output between [0,1] and Tanh between [-1,1]. so, doesn't matter how much big error you are passing, it will give small value and when we keep on doing this, after some layers values become so small that change in weights are significantly small and learning stops.
* Now , as we know, weight change in one hidden layer depends upon the weights in previous layers. If weight change in previous layer is small, then it will get smaller in current layer and because of that we get either negative values of gradient or training become very slow or sometimes it stops and show error.
###### Summary: 
* <b>Gradient based learning activation functions, squash large inputs into small range. so even large inputs will produce small change in output. When we stack multiple layers, then first layer will map input to small region, second layer will map output of first layer to further small region and keeps going on. As a result, even a large change in parameters of first layer will not produce much effect on output.</b>

