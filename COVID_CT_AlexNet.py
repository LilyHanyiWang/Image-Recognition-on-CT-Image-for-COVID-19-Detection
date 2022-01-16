import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms.transforms import Normalize, RandomRotation
import numpy as np
import os 

alexnet = models.AlexNet(num_classes=2)#.cuda()
vgg = models.vgg16(num_classes=2)#.cuda()
resnet18 = models.resnet18(num_classes=2)#.cuda()


net = alexnet

# train_dataset_path = './COVIDx_CT_split/train/'
# val_dataset_path = './COVIDx_CT_split/val/'
# test_dataset_path = './COVIDx_CT_split/test/'

train_dataset_path = './COVID_CT/train/'
val_dataset_path = './COVID_CT/val/'
test_dataset_path = './COVID_CT/test/'

# 数据预处理，resize 244X244, normalize
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# 便捷化读取图片
train_dataset = datasets.ImageFolder(root = train_dataset_path,transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64,  shuffle=True)
val_dataset = datasets.ImageFolder(root = val_dataset_path,transform=transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 64,  shuffle=True)
test_dataset = datasets.ImageFolder(root = test_dataset_path,transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64,  shuffle=True)


CEL = torch.nn.CrossEntropyLoss()

opti = torch.optim.Adam(net.parameters(), lr = 0.01)

num_epoch = 60 
if not os.path.exists('./AlexNet_small/'):
    os.makedirs('./AlexNet_small/')
for epoch in range(num_epoch):
    record_acc = [] # 记录最高准确率
    epoch_loss = []
    '''train step'''
    for data in train_dataloader:
        images,label = data
        images = images#.cuda()
        label = label#.cuda()
        opti.zero_grad()
        output = net(images)
        loss = CEL(output,label)
        loss.backward()
        opti.step()
        epoch_loss.append(loss.item())
    print('Current training loss: '+ str(np.mean(epoch_loss)))
    val_loss = []
    test_acc = []
    with torch.no_grad():
        '''val step'''
        for data in val_dataloader:
            images, label = data
            images = images#.cuda()
            label = label#.cuda()
            output = net(images)
            loss = CEL(output,label)
            val_loss.append(loss.item())
        print('Current val loss: '+ str(np.mean(val_loss)))
        '''test step'''
        for data in test_dataloader:
            images, label = data
            images = images#.cuda()
            label = label#.cuda()
            output = net(images)
            pred = torch.argmax(output,1)
            acc = (pred == label).sum().item()/label.shape[0]
            test_acc.append(acc)
        record_acc.append(np.mean(test_acc))
        print('Current test accuracy: '+ str(np.mean(test_acc)) + '  Max test accuracy is ' + str(np.max(record_acc)))

    save_path = './AlexNet_small/'+str(epoch)+'.pth'
    torch.save(net.state_dict(),save_path)