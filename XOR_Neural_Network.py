import numpy as np
import Vanilla_Neural_Network

# Display XOR Dataset

# # Make XOR Truth Table into X and y sets
# X, y = np.array([[0,0],[0,1],[1,0],[1,1]]) , np.array([[0],[1],[1],[0]]) # XOR

# # Make OR Truth Table into X and y sets
# X, y = np.array([[0,0],[0,1],[1,0],[1,1]]) , np.array([[0],[1],[1],[1]]) # OR

# Make AND Truth Table into X and y sets
X, y = np.array([[0,0],[0,1],[1,0],[1,1]]) , np.array([[0],[0],[0],[1]]) # AND

import matplotlib.pyplot as plt
for point, style in zip(X, ['>','<','>','<']):
    print(point)
    plt.plot(point, style)
plt.show()

# # print(X)
# # print(y)
Neural_Network = Vanilla_Neural_Network.NeuralNetwork([2,2,2,2,1], alpha=0.2)
Neural_Network.fit(X, y, epochs=20000)

for x, target in zip(X, y):
    pred = Neural_Network.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data = {}, answer = {}, prediction = {:.4f}, step = {}".format(x, target[0], pred, step))