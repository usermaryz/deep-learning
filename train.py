import numpy as np
import math
import random
nn = MLP(3, [4, 4, 1])

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

def loss_fn(prediction, target):
    return (prediction - target) ** 2

# тренировочный range
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0

    for x, y in zip(xs, ys):
        # Forward pass to get the predicted output
        prediction = nn(x)

        # вычисляем ошибку
        loss = loss_fn(prediction, y)
        total_loss += loss

        # обнуляем градиент
        nn.zero_grad()

        # Backward pass to compute gradients
        loss.backward()

        # обновляем веса
        optimizer.step()

        # высчитываем точность
        if abs(prediction.data - y) < 0.5:
            correct_predictions += 1

    # точность и средняя ошибка
    accuracy = correct_predictions / len(xs)
    average_loss = total_loss / len(xs)

    # печатаем прогресс
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}, Accuracy: {accuracy*100}%")

#другое
xx = np.linspace(-math.pi, math.pi, 2000)
# np.random.shuffle(arr)
# np.random.shuffle(xx)
yy = np.sin(xx).tolist()

random.seed(41)
a = Value(random.uniform(-1,1))
b = Value(random.uniform(-1,1))
c = Value(random.uniform(-1,1))
d = Value(random.uniform(-1,1))
parameters = [a, b, c, d]

pseudo_sin = lambda x: a + b * x  + c * x ** 2 + d * x ** 3
print(pseudo_sin)
print("sin", math.sin(a.data))

import sys
import matplotlib.pyplot as plt
print(sys.getrecursionlimit())
sys.setrecursionlimit(5000)
#%%wandb
for epoch in range(500):
  for step in range(2000):
    polynom = pseudo_sin(xx[step]) #предсказание зачение синуса
    loss = (polynom - yy[step]) ** 2
    # if loss.data < 0.01:
    #   break
    #wandb.log({'loss': loss.data})

    if step % 100 == 0:
        print(f"{step} step - loss {loss}")

    #set grad to zero
    for p in parameters:
      p.grad = 0

    #call backward
    loss.backward()

    lr = 0.0001
    #update weights
    for p in parameters:
      p.data = p.data - p.grad * lr

print(f'Result: y = {a.data} + {b.data} x + {c.data} x^2 + {d.data} x^3')



np.random.seed(1337)
random.seed(1337)

model = MLP(2, [16, 16, 1]) # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))

import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# make up a dataset

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.1) # [n x 2], [n,]
y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')