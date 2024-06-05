# %%
#Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from PIL import Image

# %%
#Transforms
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

# %%
#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.img_labels = []
        for label, class_dir in enumerate(os.listdir(img_dir)):
            class_path = os.path.join(img_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(class_path, img_name)
                        self.img_labels.append((img_path, label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# %%
train_path = r'C:\Users\skyja\Desktop\Data\scene_detection\seg_train\seg_train'
test_path = r'C:\Users\skyja\Desktop\Data\scene_detection\seg_test\seg_test'
pred_path = r'C:\Users\skyja\Desktop\Data\scene_detection\seg_pred\seg_pred'

# %%

train_dataset = CustomImageDataset(img_dir=train_path, transform=transformer)
test_dataset = CustomImageDataset(img_dir=test_path, transform=transformer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

# %%

#CNN Network
class ConvNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ConvNet,self).__init__()

        #Output size after convolution filter
        #((w-f+2P)/s) +1

        #Input shape= (256,3,150,150)

        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)

        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)


        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)



        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)


        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)



        #Feed forwad function

    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)

        output=self.pool(output)

        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)


            #Above output will be in matrix form, with shape (256,32,75,75)

        output=output.view(-1,32*75*75)


        output=self.fc(output)

        return output


# %%
def train_model():
    model = ConvNet(num_classes = 6).to(device)
    optimizer = Adam(model.parameters(), lr =0.001, weight_decay =0.001)
    loss_function = nn.CrossEntropyLoss()
    num_epochs = 10
    #calculating the size of training and testing images
    train_count=len(glob.glob(train_path+'/**/*.jpg'))
    test_count=len(glob.glob(test_path+'/**/*.jpg'))
    print(train_count,test_count)

    best_accuracy=0.0

    for epoch in range(num_epochs):
        
        #Evaluation and training on training dataset
        model.train()
        train_accuracy=0.0
        train_loss=0.0
        
        for i, (images,labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images=images.to(device)
                labels=labels.to(device)
                
            optimizer.zero_grad()
            
            outputs=model(images)
            loss=loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            
            
            train_loss+= loss.cpu().data*images.size(0)
            pred = outputs.argmax(dim=1)
            
            train_accuracy+=int(torch.sum(pred==labels.data))
            
        train_accuracy=train_accuracy/train_count
        train_loss=train_loss/train_count
        
        
        # Evaluation on testing dataset
        model.eval()
        
        test_accuracy=0.0
        for i, (images,labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images=images.to(device)
                labels=labels.to(device)
                
            outputs=model(images)
            pred = outputs.argmax(dim=1)
            test_accuracy+=int(torch.sum(pred==labels.data))
        
        test_accuracy=test_accuracy/test_count
        
        
        print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
        
        #Save the best model
        if test_accuracy>best_accuracy:
            torch.save(model.state_dict(),'best_checkpoint.model')
            best_accuracy=test_accuracy
    return model
        


# %%
if __name__ == "__main__":
    model = train_model()
    

# %%


# %%
if __name__ == "__main__":
    model.state_dict

# %%
if __name__ == "__main__":
    checkpoint=torch.load('best_checkpoint.model')
    model=ConvNet(num_classes=6)
    model.load_state_dict(checkpoint)
    model.eval()

    # %%
    #prediction function
def prediction(img_path,transformer,model,classes):
    
    image=Image.open(img_path)
    
    image_tensor=transformer(image).float()
    
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=Variable(image_tensor)
    
    
    output=model(input)
    
    index=output.data.numpy().argmax()
    
    pred=classes[index]
    
    return pred
    

# %%
if __name__ == "__main__":
    images_path=glob.glob(pred_path+'/*.jpg')

    # %%
    pred_dict={}

    for i in images_path:
        pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)

    # %%
    pred_dict

    # %%
    import itertools
    dict(itertools.islice(pred_dict.items(), 13148))

    # %%
    import itertools
    dict(itertools.islice(pred_dict.items(),1000,1100))

    # %%
    items = list(pred_dict.items())

    # %%
    items

    # %%
    items[1]

    # %%
    second_key, second_value = items[1]
    second_key

    # %%



