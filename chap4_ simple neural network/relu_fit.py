# 基于 numpy 框架的结论验证

import tqdm
import numpy as np
import matplotlib as plt

EPOCHS = 1000
lr = 1e-4

class Config:
    def __init__(self):
        self.inp_d = 10
        self.width = 128
        self.out_d = 10

class Model:
    def __init__(self, config : Config):
        self.W1 = np.random.randn([config.inp_d, config.width])
        self.b1 = np.random.randn([config.width])
        self.W2 = np.random.randn([config.inp_d, config.width])
        self.b2 = np.random.randn([config.width])

        
    def ReLU(x):
        y = x if x > 0 else 0
        return y
    
    def forword(self, x):
        h1 = np.dot(x, self.W1) + self.b1
        self.act = self.ReLU(h1)
        return np.dot(self.act, self.W2) + self.b2
    
    def backword(self, x, y, label):
        dy = 2 * np.mean((y - label), keepdims=True)
        dW2 = np.dot(self.act, dy)
        db2 = dy
        dact = np.dot(dy, self.W2.T)
        dh1 = dact * (x > 0).astype(float)
        dW1 = np.dot(x, dh1)
        db1 = dh1
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.b1 -= lr * db1
        self.W1 -= lr * dW1
    
    def loss(y, label):
        return np.mean((y - label) ** 2)
    
def train(x, y, model : Model):
    
    pass
    
def function(x):
    y = np.sin(x)
    return y

def main():
    # 准备模型
    model = Model()
    
    # 准备数据集
    x = np.linspace(-1, 1, 10000)
    y = function(x)
    x_train, y_train = x[0:2000], y[0:2000]
    x_test, y_test = x, y
    
    res = train(x_train, y_train, model)
    
    
    
    pass

if __name__ == '__main__':
    main()
    
    


