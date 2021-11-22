from torch.utils.data import DataLoader, random_split
from prepare import *
from model import *
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("logs")
device = torch.device("cuda")

lr = 0.001
epoch = 2000

def train(images, labels, model):
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    # optim = torch.optim.SGD(model.parameters(), lr, weight_decay=1e-6, momentum=0.9)

    model.train()   # 开始训练
    for i in range(epoch):
        # 每一轮重新打乱并提取数据
        train_dataset = GetDataset(images[0], labels[0])
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        train_accuracy = 0
        total__train_loss = 0
        for data in train_dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            y_prc = model(x)    # 计算预测值
            # print(y_prc.shape, y.shape)

            # 计算损失函数并后向传播
            loss = loss_fn(y_prc, y)    # 计算最后一层损失
            optim.zero_grad()   # 设定优化的方向
            loss.backward()     # 从最后一层损失反向计算到所有层的损失     
            optim.step()        # 更新权重

            train_accuracy += (y_prc.argmax(1) == y).sum()
            total__train_loss += loss.item()
        # print(train_accuracy)
        writer.add_scalar("train_acc", (float(train_accuracy)/float(len(train_dataset)))*100, i)

        # 每轮验证进行一次验证模型
        valid_dataset = GetDataset(images[1], labels[1])
        valid_dataloader = DataLoader(valid_dataset, batch_size=80, shuffle=True)
        valid_accuracy = 0
        total__valid_loss = 0
        model.eval()       # 开始验证
        with torch.no_grad():    # 不进行优化
            for data in valid_dataloader:
                x, y = data
                x = x.to(device)
                y = y.to(device)
                y_prc = model(x)    # 计算预测值
                # 计算损失函数并后向传播
                loss = loss_fn(y_prc, y)    # 计算最后一层损失   
                valid_accuracy += (y_prc.argmax(1) == y).sum()
                total__valid_loss += loss.item()
        
        print("epoch：{}；训练准确率：{:.0f}%；验证准确率：{:.0f}%".
                format(i, (float(train_accuracy)/float(len(train_dataset)))*100, 
                        (float(valid_accuracy)/float(len(valid_dataset)))*100))
        print("--------------------------------------------train_loss：{}，valid_loss：{}".format
                                            (total__train_loss, total__valid_loss))
        # print(train_accuracy, valid_accuracy)
        writer.add_scalar("valid_acc", (float(valid_accuracy)/float(len(valid_dataset)))*100, i)
    return model


def test(images, labels, model):
    test_dataset = GetDataset(images[2], labels[2])
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    test_accuracy = 0
    total__test_loss = 0

    model.eval()      # 开始测试
    with torch.no_grad():    # 不进行优化
        for data in test_dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            y_prc = model(x)    # 计算预测值
            # 计算损失函数并后向传播
            loss = loss_fn(y_prc, y)    # 计算最后一层损失   
            test_accuracy += (y_prc.argmax(1) == y).sum()
            total__test_loss += loss.item()
    
    print("测试准确率：{}%".
                            format((test_accuracy/len(test_dataset))*100))
    print("test_loss：{}".format(total__test_loss))


if __name__ == '__main__':
    images, labels = get_data("dataset")     # 获取数据
    model = Model()     # 创建模型
    model.to(device)
    summary(model, (1, 92, 112))            # 输出模型结构

    model = train(images, labels, model)    # 训练模型
    test(images, labels, model)             # 测试模型
    torch.save(model.state_dict(), "model/my_own_model.pth")    # 保存模型
    writer.close()

