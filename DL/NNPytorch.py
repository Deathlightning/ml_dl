import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 数据集
iris = load_iris()
xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
x = torch.FloatTensor(xTrain)
y = torch.LongTensor(yTrain)
# 建立模型
net = torch.nn.Sequential(
    torch.nn.Linear(4, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 3),
)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()
# 训练
for t in range(300):
    out = net(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 10 == 0:
        prediction = torch.max(out, dim=1)[1]
        pred_y = prediction.data.numpy()
        accuracy = np.sum(pred_y == np.array(y)) / float(len(y))
        print(f"train epoch:{t} accuracy:{accuracy}")
# 测试
x = torch.FloatTensor(xTest)
y = torch.LongTensor(yTest)
out = net(x)
prediction = torch.max(out, dim=1)[1]
pred_y = prediction.data.numpy()
accuracy = np.sum(pred_y == np.array(y)) / float(len(y))
print(f"test accuracy:{accuracy}")
