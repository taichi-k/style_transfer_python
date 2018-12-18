import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from torch.autograd import Variable
from torch import optim

def read_image(fname):
    img = cv2.imread(fname)
    img = cv2.resize(img, (256,256))
    img = img.astype(np.float32)
    img = img.transpose(2,0,1)
    return np.array([img])

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

def subtract_imagenet_mean_batch(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:,0,:,:] = 103.939
    mean[:,1,:,:] = 116.779
    mean[:,2,:,:] = 123.680
    batch = batch - Variable(mean, requires_grad = False)
    return batch

def add_imagenet_mean_batch(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:,0,:,:] = 103.9393
    mean[:,1,:,:] = 116.889
    mean[:,2,:,:] = 123.680
    batch = batch + Variable(mean, requires_grad = False)
    return batch

def save_image(image, fname):
    image = add_imagenet_mean_batch(image)
    image = image.clamp(0,255).data[0].numpy().transpose(1,2,0)
    image = image.astype(np.uint8)
    cv2.imwrite(fname,image)

def gram(input):
    b,c,h,w = input.size()
    F = input.view(b,c,h*w)
    G = torch.bmm(F,F.transpose(1,2))
    G.div_(h*w)
    return G

vgg = VGG()
vgg.load_state_dict(torch.load("vgg_conv.pth"))

content_name = "content.jpg"
content = read_image(content_name)
content = torch.FloatTensor(content)
content = Variable(content)
content = subtract_imagenet_mean_batch(content)

content_output = vgg(content, ["r42"])

style_name = "style.jpg"
style = read_image(style_name)
style = torch.FloatTensor(style)
style = Variable(style)
style = subtract_imagenet_mean_batch(style)

style_output = vgg(style, ['r11','r21','r31','r41','r51'])

content_target = [content_output[0].detach()]

style_target = []
for s in style_output:
    style_target.append(gram(s).detach())

style_weights = [0.244140625, 0.06103515625, 0.0152587890625, 0.003814697265625, 0.003814697265625]
    
result = Variable(content.data.clone(), requires_grad = True)


optimizer = optim.LBFGS([result])
mse = nn.MSELoss()
num = 0
while num <= 500:
    def closure():
        global num
        optimizer.zero_grad()
        result_output = vgg(result, ['r11','r21','r31','r41','r51','r42'])
        result_content_loss = mse(result_output[5],content_target[0])
        result_style_loss = 0
        for i,_ in enumerate(result_output[:-1]):
            result_style_loss += mse(gram(result_output[i]), style_target[i])*style_weights[i]
        total_loss = result_content_loss + result_style_loss
        total_loss.backward()
        print(num, total_loss)
        num += 1
        return total_loss
    optimizer.step(closure)
    save_image(torch.cat([style,content,result],2),"saved/"+str(num)+".jpg")

