import torch
from torch import nn
from d2l import torch as d2l

'''AlexNet使用是IMGNet数据集，输入为(X x 3 x 224 x 224)
    这里使用的是Fashion-MNIST数据集，输入为(X x 1 x 224 x 224)'''
net = nn.Sequential(
    #输入(X x 1 x 224 x 224)
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    #输出(X x 96 x 54 x 54)
    nn.MaxPool2d(kernel_size=3, stride=2),
    #输出(X x 96 x 26 x 26)
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    #输出(X x 256 x 26 x 26)
    nn.MaxPool2d(kernel_size=3, stride=2),
    #输出(X x 256 x 12 x 12)
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    #输出(X x 384 x 12 x 12)
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    #输出(X x 384 x 12 x 12)
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    #输出(X x 256 x 12 x 12)
    nn.MaxPool2d(kernel_size=3, stride=2),
    #输出(X x 256 x 5 x 5)
    nn.Flatten(),
    #输出(X x 6400)
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    #输出(X x 4096)
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    #输出(X x 4096)
    nn.Linear(4096, 10)
    #输出(X x 10)
)

'''三、使用GPU训练'''
def evaluate_accuracy_gpu(net, data_iter, device=None):
    #使用GPU计算模型在数据集上的精度
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            #把net.parameters()的第一个元素拿出来
            device = next(iter(net.parameters())).device
    #metric为（1xn）的累加器
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


'''使用GPU训练函数'''

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device='cpu'):
    #初始化net，w， b
    def init_weight(m):
        if type(m) == nn.Linear or type(m) ==nn.Conv2d:
            #xavier均匀分布，根据输入输出的大小，在输入的时候，随机输入和输出的方差是差不多的
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    print('training on', device)
    net.to(device)
    #定义损失函数
    loss = nn.CrossEntropyLoss()
    #定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr)
    #动画效果
    animator = d2l.Animator(xlabel= 'num_epochs', ylabel= 'loss',
                            xlim= [1, num_epochs], yscale= 'log',
                            legend= ['train_loss','train_acc','test_acc'])
    timer, num_batchs = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        mertic = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate (train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            mertic.add(l * X.shape[0], d2l.accuracy(y_hat, y), y.numel())
            timer.stop()
            train_l = mertic[0] / mertic[2]
            train_acc = mertic[1] / mertic [2]
            if (i + 1) % (num_batchs // 5) == 0 or i == num_batchs:
                animator.add((epoch + (i + 1)) / num_batchs,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc{train_acc:.3f}, test acc{test_acc:.3f}')
    print(f'{mertic[2] * num_epochs / timer.sum():.1f} examples/sec'
          f'on {str(device)}')

X = torch.randn(1, 1, 224, 224)
#打印各层网络信息
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)

batch_size = 128
#由于fashion_mnist中的数据是28x28，所以这里把图片拉成224x224
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, 1)
