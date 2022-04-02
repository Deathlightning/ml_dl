import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


def mean(X):
    """
    计算均值
    :param X:一维向量
    :return:
    """
    return sum(X) / float(len(X))


def stDev(x):
    """
    计算标准差
    :param x:一维向量
    :return:
    """
    avg = mean(x)
    return np.sqrt(sum([np.power(i - avg, 2) for i in x]) / len(x))


def gaussianProbability(x, mean, stDev):
    """
    计算高斯分布
    :param x:
    :param mean:
    :param stDev:
    :return:
    """
    ex = np.exp(-(pow(x - mean, 2) / (2 * pow(stDev, 2))))
    return (1 / np.sqrt(2 * np.pi * pow(stDev, 2))) * ex


class NaiveBayes:
    def __init__(self):
        # 不同类别下，各特征的mean和标准差
        self.model = {}
        # 先验概率
        self.priorProbability = {}

    def calPriorProbability(self, yList):
        """
        先验概率
        :param yList:
        :return:
        """
        a = dict(Counter(yList))
        for k, v in a.items():
            a[k] = v / len(yList)
        self.priorProbability = a

    @staticmethod
    def summarize(train_data):
        """
        获取每个特征的均值和方差
        :param train_data:二维list，是某一类别下的全部样本
        :return:4*2的元组，一行表示一个特征的平均值和方差
        """
        # 二维list前加*表示打散为多个一维list
        # 用zip可以同时遍历这些list，也就是按列遍历，i就是全部样本下某特征的全部取值
        return [(mean(i), stDev(i)) for i in zip(*train_data)]

    def fit(self, x, y):
        """
        求出现有数据各特征均值和标准差
        :param x:
        :param y:
        :return:
        """
        self.calPriorProbability(y)
        labels = set(y)
        # 初始化概率dict
        data: dict = {label: [] for label in labels}
        for xItem, label in zip(x, y):
            data[label].append(xItem)
        self.model: dict = {label: self.summarize(value) for label, value in data.items()}

    def calculateProbabilities(self, testSample):
        """
        计算贝叶斯公式的分子，及先验概率与似然函数的乘积
        :param testSample:单个样本数据
        :return:返回各样本的似然函数乘积
        """
        # input_data:[1.1,2.2,0.5]
        probabilities = {}
        for label, value in self.model.items():
            # 用先验概率初始化结果概率
            probabilities[label] = self.priorProbability[label]
            for i in range(len(value)):
                mean, stDev = value[i]
                probabilities[label] *= gaussianProbability(testSample[i], mean, stDev)
        return probabilities

    def predict(self, x_test):
        """
        新样本预测
        此处按升序排列，则取-1指的是找最大似然概率，取0指取出所属类别
        :param x_test:测试样本
        :return:label
        """
        return sorted(self.calculateProbabilities(x_test).items(), key=lambda x: x[-1])[-1][0]

    def score(self, x_test, y_test):
        """
        计算准确率
        :param x_test:
        :param y_test:
        :return:
        """
        right = 0
        for x, y in zip(x_test, y_test):
            if self.predict(x) == y:
                right += 1
        return right / float(len(x_test))


def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
    model = NaiveBayes()
    model.fit(X_train, y_train)
    # 预测
    print(f"测试单个样本所属类别：{model.predict([4.4, 3.2, 1.3, 0.2])}")
    # 计算准确率ed
    print(model.score(X_test, y_test))


main()
