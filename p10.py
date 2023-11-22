import matplotlib.pyplot as plt 
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()

# Dense Layer
class Layer_Dense: 

	# Layer initialization
	def __init__(self, n_inputs, n_neurons): 
		# Shape - need to know size of input and number of neurons
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)	
		self.biases = np.zeros((1, n_neurons))

	# Forward pass
	def forward(self, inputs): 
		# Remember input values
		self.inputs = inputs 
		# Calculate output values from inputs, weights and biases
		self.output = np.dot(inputs, self.weights) + self.biases

	# Backward pass
	def backward(self, dvalues):
		# Gradients on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
		# Gradient on values 
		self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU: 

	# Forward pass
	def forward(self, inputs):
		# Remember input values 
		self.inputs = inputs 
		# Calculate output values from inputs
		self.output = np.maximum(0, inputs)
	
	# Backward pass
	def backward(self, dvalues):
		# Since we need to modify the original var, let's copy the values first
		self.dinputs = dvalues.copy()

		# Zero gradient where input values were negative
		self.dinputs[self.inputs <= 0] = 0 

# Softmax activation
class Activation_Softmax: 
	# Takes a vector of numerical values as input
	# amd transforms them into a probability distribution
	# which makes it easier to interpret 

	# Forward pass
	def forward(self, inputs):
		self.inputs = inputs 
		# Remember input values 
		self.inputs = inputs 

		# Get unnormalized probabilities
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		# This operation effectively centers the values in inputs around their maximum value in each row.
		# This is done for numerical stability, as exponentiating very large or very small numbers can lead to numerical inaccuracies.

		# Normalize probabilities for each sample
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

		self.output = probabilities

	# Backward pass
	def backward(self, dvalues):

		# Create uninitialized array 
		self.dinputs = np.empty_like(dvalues)

		# Enumerate outputs and gradients 
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			# Flatten output array into a column vector
			single_output = single_output.reshape(-1, 1)

			# Calculate Jacobian matrix of the output 
			# Jacobian represents the rate at which each output element
			# changes with respect to changes in the input elements
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

			# Calculate sample-wise gradient & add to array of sample gradients
			# Computes the gradients w.r.t the inputs in the softmax function 
			# matrix x between Jacobian and the gradients received from the subsequent layer
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Common loss class
class Loss: 

	# Calculates the data and regularization losses 
	# given model output and ground truth values
	def calculate(self, output, y):
		# y represents target values

		# Calculate sample losses
		sample_losses = self.forward(output, y)

		# Calculate mean loss
		data_loss = np.mean(sample_losses)

		# Return loss
		return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

	# Forward pass
	def forward(self, y_pred, y_true):

		# number of samples in a batch
		samples = len(y_pred)

		# Clip data to prevent division by 0
		# Clip both sides to not drag mean towards any value 
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

		# Probabilities for target values

		# handling 1-dimensional array (categorical labels)
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]

		# handling 2-dimensional array (one-hot encoded labels)
		elif len(y_true.shape) ==2:
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

		# Losses 
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	# Backward pass
	def backward(self, y_pred, y_true):

		# Number of samples 
		samples = len(y_pred)
		# Number of labels in every sample
		# We'll use the first sample to count them
		labels = len(y_pred[0])

		# If labels are spares, turn them into one-hot vector
		if len(y_true.shape) ==1:
			y_true = np.eye(labels)[y_true]

		# Calculate gradient 
		self.dinputs = -y_true/y_pred
		# Normalize gradient 
		self.dinputs = self.dinputs/samples


# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
# Softmax combined with Categorical Cross-Entropy loss is often used 
class Activation_Softmax_Loss_CategoricalCrossentropy:

	# Creates activation and loss function objects
	def __init__(self):
		self.activation = Activation_Softmax()
		self.loss = Loss_CategoricalCrossentropy()

	# Forward pass 
	def forward(self, inputs, y_true):
		# Output layer's activation function 
		self.activation.forward(inputs)
		# Set the output 
		self.output = self.activation.output 
		# Calculate and return loss value 
		return self.loss.calculate(self.output, y_true)

	# Backward pass
	def backward(self, y_pred, y_true):

		# Number of samples
		samples = len(y_pred)

		# If labels are one-hot encoded, turn them into discrete values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)

		# Copy so we can safely modify 
		self.dinputs = y_pred.copy()

		# Calculate gradient 
		# for each sample, subtract 1 from predicted value corresponding to the true class label
		# from the derivative of the categorical cross-entropy loss w.r.t predicted values
		self.dinputs[range(samples), y_true] -= 1
		
		# Normalize gradient 
		self.dinputs = self.dinputs / samples 


# SGD Optimizer
class Optimizer_SGD:

	# Initialize optimizer - set settings
	# learning rate of 1 is default for this optimizer
	def __init__(self, learning_rate=1.0):
		self.learning_rate = learning_rate

	# Update parameters
	def update_params(self, layer):
		layer.weights += -self.learning_rate * layer.dweights
		layer.biases += -self.learning_rate * layer.dbiases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

#Create DENSE layer with 2 input features and 3 output values 
dense1 = Layer_Dense(2, 3)
#Create ReLU ACTIVATION function (to be used with Dense layer): 
activation1 = Activation_ReLU()
#Create second DENSE layer w/ 3 input features (as we take output of previous layer here and 3 output values)
dense2 = Layer_Dense(3, 3)
#Create Softmax classifier's combined LOSS and ACTIVATION
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through DENSE layer 1
dense1.forward(X)

# Perform a forward pass through ACTIVATION function
# takes the output of the first DENSE layer
activation1.forward(dense1.output)

# Perform a forward pass through DENSE layer 2 
# takes the output of first layer ACTIVATION function
dense2.forward(activation1.output)

# Perform a forward pass through the ACTIVATION/LOSS combined function
# takes the output of second DENSE layer and returns LOSS
loss = loss_activation.forward(dense2.output, y)

# Let's see output of the first few samples: 
print(loss_activation.output[:5])

# Print loss value 
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets 
# Calculate values along first axis 
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
	y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

# Print accuracy
print('acc:', accuracy)

# Backward pass 
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

optimizer = Optimizer_SGD()
optimizer.update_params(dense1)
optimizer.update_params(dense2)




