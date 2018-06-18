from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch import nn,optim
import torchvision.transforms as transforms
import json
import os
import PIL
from time import strftime
num_epochs = 100
batch_size = 64
learning_rate = 0.001

#自定义数据集
class mydataset(torch.utils.data.Dataset):
        def __init__(self,dir,annotation,train):
                with open(annotation) as f:
                        datas = json.load(f)
                self.dir = dir
                self.images = [i["image_id"] for i in datas]
                self.labels = [int(i["label_id"]) for i in datas]
                self.train = train
                self.val_transform = transforms.Compose([
                        transforms.Scale(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
                        ])
                self.train_transform = transforms.Compose([
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
                        ])
        
        def __getitem__(self,index):
                image_path = os.path.join(self.dir,self.images[index])
                img = PIL.Image.open(image_path).convert('RGB')
                if self.train:
                        img = self.train_transform(img)
                else:
                        img = self.val_transform(img)
                return img,self.labels[index]
        
        def __len__(self):
                return len(self.images)

#说明输入数据的路径
train_images_path = "/home/szh/AIchallenger/ai_challenger_scene_train_20170904/scene_train_images_20170904"
train_annotation_path = "/home/szh/AIchallenger/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json"
traindataset = mydataset(train_images_path,train_annotation_path,True)
trainloader = torch.utils.data.DataLoader(traindataset,batch_size,shuffle=True,num_workers=4)
val_images_path = "/home/szh/AIchallenger/ai_challenger_scene_validation_20170908/scene_validation_images_20170908"
val_annotation_path = "/home/szh/AIchallenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json"
valdataset = mydataset(val_images_path,val_annotation_path,False)
valloader = torch.utils.data.DataLoader(valdataset,batch_size,shuffle=False,num_workers=4)
test_images_path = "/home/szh/AIchallenger/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922"
test_annotation_path = "/home/szh/AIchallenger/ai_challenger_scene_test_a_20170922/ai_challenger_scene_test_a_20170922.json"
testdataset = mydataset(test_images_path,test_annotation_path,False)
testloader = torch.utils.data.DataLoader(testdataset,batch_size,shuffle=False,num_workers=4)

#计算topk正确的个数
def topk_correct(score,label,k=3):
        topk = score.topk(k)[1]
        label = label.view(-1,1).expand_as(topk)
        correct = (label==topk).float().sum()
        return correct

#格式化指定字符串
def getinfo(epoch,lr,train_top1,train_top3,val_top1,val_top3):
        return strftime('[%m%d_%H%M%S]')+('epoch:{epoch},lr:{lr},train_top1:{train_top1},'
        'train_top3:{train_top3},val_top1:{val_top1},val_top3:{val_top3}').format(
        epoch=epoch,
        lr=lr,
        train_top1=train_top1,
        train_top3=train_top3,
        val_top1=val_top1,
        val_top3=val_top3)

#模型定义

def conv3x3(in_channels,out_channels,stride=1):
        return nn.Conv2d(in_channels,out_channels,kernel_size=3,
                stride=stride,padding=1,bias=False)
class ResidualBlock(nn.Module):
        def __init__(self,in_channels,out_channels,stride=1,downsample=None):
                super(ResidualBlock,self).__init__()
                self.conv1 = conv3x3(in_channels,out_channels,stride)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(out_channels,out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.downsample = downsample
        
        def forward(self,x):
                residual = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample:
                        residual = self.downsample(x)
                out += residual
                out = self.relu(out)
                return out

class ResNet(nn.Module):
        def __init__(self,block,layers,num_classes=80):
                super(ResNet,self).__init__()
                self.in_channels =16 
                self.conv = conv3x3(3,16)
                self.bn = nn.BatchNorm2d(16)
                self.relu = nn.ReLU(inplace=True)
                self.layer1 = self.make_layer(block,16,layers[0])
                self.layer2 = self.make_layer(block,32,layers[1],2)
                self.layer3 = self.make_layer(block,64,layers[2],2)
                self.avg_pool = nn.AvgPool2d(8)
                self.fc = nn.Linear(3136,num_classes)
        
        def make_layer(self,block,out_channels,blocks,stride=1):
                downsample = None
                if (stride != 1) or (self.in_channels != out_channels):
                        downsample = nn.Sequential(
                                conv3x3(self.in_channels,out_channels,stride=stride),
                                nn.BatchNorm2d(out_channels))
                layers = []
                layers.append(block(self.in_channels,out_channels,stride,downsample))
                self.in_channels = out_channels
                for i in range(1,blocks):
                        layers.append(block(out_channels,out_channels))
                return nn.Sequential(*layers)
        
        def forward(self,x):
                out = self.conv(x)
                out = self.bn(out)
                out = self.relu(out)
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.avg_pool(out)
                out = out.view(out.size(0),-1)
                out = self.fc(out)
                return out
                
resnet = ResNet(ResidualBlock,[3,3,3])
resnet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(),lr=learning_rate)
bestaccurcy = 0
for epoch in range(num_epochs):
        
        total = 0
        correct3 = 0
        correct = 0
        lossdata=0
        #训练
        resnet.train()
        for images,labels in tqdm(trainloader):
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                
                optimizer.zero_grad()
                outputs = resnet(images)
                loss = criterion(outputs,labels)
                total += labels.size(0)
                correct3 += topk_correct(outputs.data,labels.data)
                correct += topk_correct(outputs.data,labels.data,k=1)
                lossdata+=loss.data[0]*labels.size(0)
                loss.backward()
                optimizer.step()
        print("Epoch [%d/%d],Loss: %.4f"%(epoch+1,num_epochs,lossdata*1.0/total)) 
        top3accuracy = correct3*1.0/total
        accuracy = correct*1.0/total
        train_top1=accuracy
        train_top3=top3accuracy
        print("top3 Accuracy and accuracy of the model on the train set are [%.4f|%.4f]"
        %(top3accuracy,accuracy))
        total = 0
        correct3 = 0
        correct = 0
        #验证
        resnet.eval()
        for images,labels in tqdm(valloader):
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                outputs = resnet(images)
                total += labels.size(0)
                correct3 += topk_correct(outputs.data,labels.data)
                correct += topk_correct(outputs.data,labels.data,k=1)
        top3accuracy = correct3*1.0/total
        accuracy = correct*1.0/total
        val_top1=accuracy
        val_top3=top3accuracy
        print("top3 Accuracy and accuracy of the model on the validation set are [%.4f|%.4f]"
        %(top3accuracy,accuracy))
        torch.save(resnet.state_dict(),getinfo(epoch,learning_rate,train_top1,train_top3,val_top1,val_top3)
                       +"resnet.pkl")
        if accuracy>bestaccurcy:
            bestaccurcy = accuracy
            torch.save(resnet.state_dict(),"bestresnet.pkl")
        if (epoch+1)%10==0:
            learning_rate /= 3
            optimizer = torch.optim.Adam(resnet.parameters(),lr=learning_rate)

resnet.load_state_dict(torch.load("bestresnet.pkl"))
total = 0
correct3 = 0
correct = 0
#测试
for images,labels in tqdm(testloader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = resnet(images)
        total += labels.size(0)
        correct3 += topk_correct(outputs.data,labels.data)
        correct += topk_correct(outputs.data,labels.data,k=1)
top3accuracy = correct3*1.0/total
accuracy = correct*1.0/total
print("top3 Accuracy and accuracy of the model on the test set are [%.4f|%.4f]"
                %(top3accuracy,accuracy))