import numpy as np

class NeuralNetwork:
    '''
    Vanilla Neural Network Implementation:
    Neural Network that can follow the standard feed-forward  architecture

    Example of a standard feed-forward architecture shorthand will use
    a list of integers to represent the network architecture
    64 - 32 - 1 is a network with 64 input neurons, 32 hidden layer 1 neurons and 1 output neuron
    '''
    def __init__(self, network_architecture, alpha=0.1):
        self.W = []
        self.layers = network_architecture
        self.alpha = alpha

        # Iterate through all the layers except the nth hidden layer and the output layer
        for i in np.arange(0, len(self.layers)-2):
            # Create weights for input -> hidden layer 1 to hidden layer n-2 to hidden layer n-1
            # works for arbritrary number of hidden layers
            w = np.random.randn(self.layers[i]+1, self.layers[i+1]+1)

            # Append the weights between different layers  (input -> h1) to (hn-2 -> hn-1)
            # Normalize the weights of each layer
            self.W.append(w/np.sqrt(self.layers[i]))
        # initializing the random weights for nth hidden layer to output
        # Append and normalize the weights of each layer
        w = np.random.randn(self.layers[-2]+1, self.layers[-1])
        self.W.append(w/np.sqrt(self.layers[-2]))

    def __repr__(self):
        # This provides the user with the shorthand notation of the neural network being generated
        return "Neural Network: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x, derive=False):
        # The sigmoid activation function that returns sigmoid result or
        # derivative result based on derive boolean flag
        S = 1.0/(1.0 + np.exp(-x))
        if derive:
            return S * (1 - S)
        return S

    #TODO need to finish the network architecture for forward feed network (I.e. need fit, predict and loss methods)
    def fit(self, X, y, epochs = 100, displayUpdate = 100):
        '''

        :param X: Dataset without labels
        :param y: Labels from Dataset
        :param epochs: Number of times the model views and trains over dataset
        :param displayUpdate: Print model training results every (displayUpdate) number of times
        :return: the models aka self
        '''

        # Bias term is added the end of our dataset (X)
        X = np.c_[X, np.ones(X.shape[0])]


        for epoch in np.arange(0,epochs):
            #pulls out a training set instance with it's respective label
            for x, target in zip(X, y):
                # pass in each instance (data and label) to function that deals with forward & backward's prop
                self.fit_partial(x,target)
            # This prints the models current results accuracy on dataset every kth epoch
            if epoch == 0 or epoch % displayUpdate == 0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch = {}, loss = {:.7f}".format(epoch+1, loss))

    def fit_partial(self, x, y):
        """
        fit partial is responsible for forward propagation predicts and updating the weights via backpropagation
        :param x: data point from X
        :param y: respective target for datapoint x
        :return:
        """
        # Check to see if the datapoint passed into the list is at least 2 dimensional
        # A is representation of results between layers in the neural network
        A = [np.atleast_2d(x)] # only x populates A right now

        # With this code below the Network is now able to provide predictions on the dataset
        # But the network is still unable to learn and update it's weight we need Backpropagation to do this

        # TODO make the Backpropagtion for the Neural Network and implement Prediction and Calculate Loss methods
        #Feed-Forward Prediction without Weight Update(i.e. Backpropagation)
        for layer in np.arange(0, len(self.W)):
            #net inpt = wTx for each respective layer passed into the network architecture
            net_input = A[layer].dot(self.W[layer])
            layer_outputs = self.sigmoid(net_input)
            A.append(layer_outputs)

        #Backpropagate through the entire network and update the weights in a feedback loop that help the Neural Network General well

        # The difference between the model output and the actual label
        error = A[-1] - y
        #Apply the chain rule and build of list of partial derivatives wrt to the error/loss
        D = [error * self.sigmoid(A[-1], True)] # Currently D only has the partial derivative for the output of the network

        # We want to start from the n-1th hidden layer ignoring the output(A[-1]) and the Nth Hidden Layer(A[-2]) that are taken into account already
        # We will calculate D for the remaining hidden layers from n-1 to the input layer
        for layer in np.arange(len(A)-2, 0, -1):
            # pass in the p. derivative of the previous layer and dot it with the weights of the current layer
            error_wrt_previous_delta = D[-1].dot(self.W[layer].T)
            #Same calculation done for D to calculate the partial derivative
            delta = error_wrt_previous_delta * self.sigmoid(A[layer], True)
            D.append(delta)
        #Need to flip the list of partial derivatives to appy sequentially to weights  b/c we calculated the p.derives from output to input (in reverse)
        D = D[::-1]

        # Finally we update the weights by applying weights at layer i +=(- eta * (output_at_layer_i o partial derivative wrt loss at layer i))
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * (A[layer].T.dot(D[layer]))
    # The forward and backpropagation is done

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones(p.shape[0])]
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    def calculate_loss(self, X, Y):
        targets = np.atleast_2d(Y)
        predictions = self.predict(X,addBias=False)
        loss = 0.5 * np.sum((targets-predictions)**2)
        return loss
