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

path = '/Users/hanyiwang/Desktop/ImageRecognitiononCTImageforCOVID-19Detection/code/AlexNet_small/'

train_dataset_path = './COVID_CT/train/'
val_dataset_path = './COVID_CT/val/'
test_dataset_path = './COVID_CT/test/'

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


test_dataset = datasets.ImageFolder(root = test_dataset_path,transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64,  shuffle=True)

num_epoch = 60 

for epoch in range(num_epoch):
    net.load_state_dict(torch.load(path+str(epoch)+'.pth'))
    record_acc = []
    epoch_loss = []
    test_acc = []
    with torch.no_grad():
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

    # save_path = './AlexNet_small/'+str(epoch)+'.pth'
    # torch.save(net.state_dict(),save_path)
