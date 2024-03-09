'''from stochaster import tensor

class StochasterBobNet:
  def __init__(self):
    self.l1 = tensor(layer_init_uniform(784, 128))
    self.l2 = tensor(layer_init_uniform(128, 10))
    return

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()
  

model = StochasterBobNet()

from stochaster.optim import SGD
from stochaster.nn import BCELoss

loss_function = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=0.15)

accuracies, losses = [], []

for i in (t := trange(1000)):
    model.train()
    y_res = model(X_train)
    
    # results are between 0 and 1 because the last layer is sigmoid
    # we round it to get classifications
    accuracy = (y_res.round() == y_train).float().mean()
    loss = loss_function(y_res, y_train)

    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracies.append(accuracy.item())
    losses.append(loss.item())
    t.set_description("accuracy %.2f loss %.2f" % (accuracy.item(), loss.item()))

plt.ylim(-0.1, 1.1)
plt.plot(losses)
plt.plot(accuracies)
'''