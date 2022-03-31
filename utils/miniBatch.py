import numpy as np
from sklearn.datasets import load_iris


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    输入：X的维度是（n,m），n是样本数，m是每个样本的特征数
    """
    np.random.seed(seed)
    n = X.shape[0]
    mini_batches = []
    # step1：打乱训练集
    xy = np.concatenate([X, Y], axis=1)
    np.random.shuffle(xy)
    x = xy[:, :xy.shape[1] - 1]
    y = xy[:, -1]
    # step2：按照batchsize分割训练集
    # 得到总的子集数目，math.floor表示向下取整
    num_complete_minibatches = int(np.floor(n / mini_batch_size))
    for k in range(num_complete_minibatches):
        # 冒号：表示取所有行，第二个参数a：b表示取第a列到b-1列，不包括b
        mini_batch_X = x[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch_Y = y[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    # m%mini_batch_size != 0表示还有剩余的不够一个batch大小，把剩下的作为一个batch
    if n % mini_batch_size != 0:
        mini_batch_X = x[mini_batch_size * num_complete_minibatches:, :]
        mini_batch_Y = y[mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    np.random.seed(None)
    return mini_batches


if __name__ == '__main__':
    iris = load_iris()
    batches = random_mini_batches(iris.data, np.resize(iris.target, (len(iris.target), 1)), 12, 10)
    for batch in batches:
        for x, y in zip(batch[0], batch[1]):
            print(x, y)
