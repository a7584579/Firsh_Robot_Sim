import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(torch.version)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        #6 kernels with different parameters to get different fetures, measns the
        #kernels' parameters are adjustable
        #weight param:in_channels × out_channels × kernel_height × kernel_width
        #1 × 6 × 5 × 5 = 150
        #bias param:out_channels = 6

        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input1 = torch.randn(1, 1, 32, 32)
out = net(input1)
print(out)

'''probs = torch.softmax(out, dim=1)  # 沿类别维度（dim=1）归一化
print(probs)'''

net.zero_grad()
out.backward(torch.randn(1, 10))

target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()#均方误差
#F.cross_entropy(out, target)#cross entropy Loss

loss = criterion(out, target)
#loss = F.cross_entropy(out, target)
#print(loss)

print(loss.grad_fn)  # MSELoss
#损失由 nn.MSELoss() 计算，其反向传播操作为 MseLossBackward
print(loss.grad_fn.next_functions[0][0])  # Linear
#MseLossBackward 的输入是 out，而 out 是通过全连接层（nn.Linear）的矩阵乘法（Addmm）生成的
#AddmmBackward0 object
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()          # 自动计算梯度（无需手动传参）
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


