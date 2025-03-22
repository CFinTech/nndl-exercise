# 基于 numpy 框架的结论验证

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCHS = 5000
lr = 1e-3

class Config:
    def __init__(self):
        self.inp_d = 1
        self.width = 64
        self.out_d = 1

class Model:
    def __init__(self, config : Config):
        self.W1 = np.random.randn(config.inp_d, config.width)
        self.b1 = np.zeros([1, config.width])
        self.W2 = np.random.randn(config.width, config.out_d)
        self.b2 = np.zeros([1, config.out_d])

    def ReLU(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        self.h1 = np.dot(x, self.W1) + self.b1
        self.act = self.ReLU(self.h1)
        return np.dot(self.act, self.W2) + self.b2
    
    def backward(self, x, y, label):
        dy = 2 * (y - label) / x.shape[0]
        dW2 = np.dot(self.act.T, dy)
        db2 = np.sum(dy, axis=0, keepdims=True)
        dact = np.dot(dy, self.W2.T)
        dh1 = dact * (self.h1 > 0).astype(float)
        dW1 = np.dot(x.T, dh1)
        db1 = np.sum(dh1, axis=0, keepdims=True)
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.b1 -= lr * db1
        self.W1 -= lr * dW1
    
    def loss(y, label):
        return np.mean((y - label) ** 2)
    
def train(x_train, y_train, x_test, y_test, model : Model):
    train_losses, test_losses = [], []
    for epoch in tqdm(range(EPOCHS)):
        y = model.forward(x_train)
        train_loss = Model.loss(y, y_train)
        model.backward(x_train, y, y_train)
        
        yt = model.forward(x_test)
        test_loss = Model.loss(yt, y_test)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 500 == 0:
            print(f"Epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}")
        
    return train_losses, test_losses, yt
    
def function(x):
    # y = np.sin(x)
    y = x ** 3 + 12
    return y

def main():
    # 准备模型
    config = Config()
    model = Model(config)
    
    # 准备数据集
    x = np.linspace(-5, 5, 10000).reshape(-1, 1)
    y = function(x)
    x_train, y_train = x[0:10000], y[0:10000]
    x_test, y_test = x, y
    
    train_losses, test_losses, yt = train(x_train, y_train, x_test, y_test, model)
    
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss.jpg")
    
    plt.figure()
    plt.plot(x_test, y_test, label="Original Function")
    plt.plot(x_test, yt, label="Prediction")
    plt.legend()
    plt.title("Function Fitness")
    plt.savefig("fit.jpg")

if __name__ == '__main__':
    main()
    
    


