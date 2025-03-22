# 基于 numpy 框架的结论验证

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置超参数
EPOCHS = 2000
lr = 1e-3
BATCH_SIZE = 32

# Xavier 初始化方式
def xavier(shape):
    return np.random.randn(*shape) * np.sqrt(2 / sum(shape))

# 模型配置
class Config:
    def __init__(self):
        self.inp_d = 1
        self.width = 64
        self.out_d = 1

# 模型定义，使用了 2 层的 ReLU 网络
class Model:
    def __init__(self, config : Config):
        self.W1 = xavier((config.inp_d, config.width))
        self.b1 = np.zeros([1, config.width])
        self.W2 = xavier((config.width, config.out_d))
        self.b2 = np.zeros([1, config.out_d])

    # 激活函数定义
    def ReLU(self, x):
        return np.maximum(0, x)
    
    # 前向传播
    def forward(self, x):
        self.h1 = np.dot(x, self.W1) + self.b1
        self.act = self.ReLU(self.h1)
        return np.dot(self.act, self.W2) + self.b2
    
    # 反向传播
    def backward(self, x, y, label):
        dy = 2 * (y - label) / x.shape[0]
        dW2 = np.dot(self.act.T, dy)
        db2 = np.sum(dy, axis=0, keepdims=True)
        dact = np.dot(dy, self.W2.T)
        dh1 = dact * (self.h1 > 0)
        grad_norm = np.linalg.norm(dh1)
        if grad_norm > 1:
            dh1 /= grad_norm
        dW1 = np.dot(x.T, dh1)
        db1 = np.sum(dh1, axis=0, keepdims=True)
        
        # 更新参数的值
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.b1 -= lr * db1
        self.W1 -= lr * dW1
    
    # 损失函数定义
    def loss(y, label):
        return np.mean((y - label) ** 2)
    
# 训练过程
def train(x_train, y_train, x_test, y_test, model : Model):
    train_losses, test_losses = [], []
    for epoch in tqdm(range(EPOCHS)):
        
        # 对于一个批次的数据
        for i in range(0, len(x_train), BATCH_SIZE):
            xb = x_train[i:i+BATCH_SIZE]
            yb = y_train[i:i+BATCH_SIZE]
            pred = model.forward(xb)
            model.backward(xb, pred, yb)
            
        test_loss = Model.loss(model.forward(x_test), y_test)
        train_loss = Model.loss(model.forward(x_train), y_train)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # 打印损失的中间值
        if epoch % 200 == 0:
            print(f"Epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}")
        
    return train_losses, test_losses, model.forward(x_test)
    
# 需要预测的函数定义
def function(x):
    y =  x ** 2 - 3 * x
    return y

def main():
    # 准备模型
    config = Config()
    model = Model(config)
    
    # 准备数据集
    x = np.linspace(-1, 1, 20000).reshape(-1, 1)
    y = function(x)
    
    # 打乱数据并划分
    idx = np.random.permutation(len(x))
    xs, ys = x[idx], y[idx]
    train_ratio = 0.8
    split = int(len(xs) * train_ratio)
    x_train, y_train = xs[:split], ys[:split]
    x_test,  y_test  = xs[split:], ys[split:]
    
    # 训练与测试
    train_losses, test_losses, yt = train(x_train, y_train, x_test, y_test, model)
    
    # 可视化损失值
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss.jpg")
    
    # 重新对数据排序
    order = np.argsort(x_test.flatten())
    x_plot = x_test[order]
    y_true = y_test[order]
    y_pred = yt[order]

    # 可视化数据
    plt.figure()
    plt.plot(x_plot, y_true, label="Original Function", linewidth=8, color=(0.7, 0.7, 0.7))
    plt.plot(x_plot, y_pred, label="Prediction", linewidth=1, color="red")
    plt.legend(); 
    plt.savefig("fit.jpg")

if __name__ == '__main__':
    main()
    
    


