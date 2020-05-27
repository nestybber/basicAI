import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import torch
import cv2

import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm_notebook

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32

# Helper function to plot a decision boundary.
def plot_decision_boundary(pytorch_model, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    pred_func = lambda x: get_pred(pytorch_model, x)
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)
    
    

def get_pred(pytorch_model, x):
    x = torch.Tensor(x).to(device)
    output = pytorch_model(x)
    output = torch.argmax(output, axis=1)
    output = output.cpu().numpy()
    return output


def get_accuracy(pred, label):
    return torch.sum(pred == label).item() / len(label)


def get_prediction(output):
    return torch.argmax(output, axis=1)


basic_block = {
    'type0': np.array(
        [[0,0,0,0],
         [0,1,1,0],
         [0,1,1,0],
         [0,0,0,0]]
    ),
    'type1': np.array(
        [[0,0,0,0],
         [0,1,0,0],
         [1,1,1,0],
         [0,0,0,0]]
    ),
    'type2': np.array(
        [[0,1,0,0],
         [0,1,1,0],
         [0,0,1,0],
         [0,0,0,0]]
    ),
    'type3': np.array(
        [[0,0,1,0],
         [0,1,1,0],
         [0,1,0,0],
         [0,0,0,0]]
    ),
    'type4': np.array(
        [[0,1,0,0],
         [0,1,0,0],
         [0,1,0,0],
         [0,1,0,0]]
    ),
    'type5': np.array(
        [[0,0,0,0],
         [0,1,0,0],
         [0,1,0,0],
         [0,1,1,0]]
    ),
    'type6': np.array(
        [[0,0,0,0],
         [0,0,1,0],
         [0,0,1,0],
         [0,1,1,0]]
    )
}


def make_tetris_dataset(N, batch_size, random_position, 
                        random_scale, random_rotation):
    images = []
    labels = []
    block_type_num = len(basic_block)
    for i in range(N//batch_size):
        image_batch = []
        label_batch = []
        for j in range(batch_size):
            block_type = np.random.choice(block_type_num)
            image, label = get_tetris_sample(img_size, block_type,
                                             random_position=random_position,
                                             random_scale=random_scale,
                                             random_rotation=random_rotation)
            image_batch.append(image)
            label_batch.append(label)
        images.append(image_batch)
        labels.append(label_batch)
        
    images = np.array(images).reshape(N//batch_size, batch_size, img_size, img_size)
    labels = np.array(labels).reshape(N//batch_size, batch_size)
    return images, labels


def get_tetris_sample(img_size, block_type, 
                      random_position=True, random_scale=False, 
                      random_rotation=False):
    if random_scale:
        block_size = np.random.choice([4, 8, 12, 16])
    else:
        block_size = 12
    block, label = get_block_image(block_size, block_type)
    if random_rotation:
        k = np.random.randint(4)
        block = np.rot90(block, k)
    if random_position:
        x, y = get_random_position(img_size, block_size)
    else:
        mid = (img_size-block_size)//2
        x, y = mid, mid
    data = np.zeros((img_size, img_size), dtype=np.uint8)
    data[y:y+block_size, x:x+block_size] = block
    return data, label
    

def get_block_image(block_size, block_type):
    
    block = basic_block['type%d'%block_type]
    block = convert_to_block_image(block)
    block = resize_block(block, block_size)
    return block, block_type


def get_random_position(img_size, block_size):
    assert img_size > block_size
    x = np.random.choice(img_size - block_size)
    y = np.random.choice(img_size - block_size)
    return x, y


def convert_to_block_image(block):
    block = Image.fromarray(np.array(block*255, dtype=np.uint8))
    return block


def resize_block(block, block_size):
    block = block.resize((block_size, block_size))
    return block


def visualize_dataset(dataset, num):
    plt.figure(figsize=(num,3))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(dataset[0][i][0])
        plt.axis('off')
        plt.title('type %d'%dataset[1][i][0])
    
    
class TetrisMLP(nn.Module):
    def __init__(self):
        super().__init__()
        ## Fill In Your Code Here ##
        self.fc1 = nn.Linear(32, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 7)
        ############################


    def forward(self, x):
        x = x.view(-1, img_size * img_size)
        ## Fill In Your Code Here ##
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        ############################
        return x
    
    
def train(model, optimizer, criterion, train_dataset, total_epoch):
    tbar = tqdm_notebook(range(total_epoch), total=total_epoch)

    # set the model to train-mode. 
    model.train()
    for epoch in tbar:
        for image_batch, label_batch in zip(*train_dataset):
            # conver numpy to torch.tensor type
            image_batch = torch.Tensor(image_batch) / 255.
            label_batch = torch.LongTensor(label_batch)

            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            loss = 0
            ## Fill In Your Code Here ##
            # step 1 to 5
            optimizer.zero_grad()
            output = model(image_batch)
            loss = ceriterion(output, label_batch)
            loss.backward()
            optimizer.step()
            ############################
            
        # print the progress
        if epoch % 10 == 9:
            pred = get_prediction(output)
            train_accuracy = get_accuracy(pred, label_batch)
            desc = '%d-th epoch, loss: %.4f, train accuracy: %.2f'%(epoch+1, 
                                                loss.item(), train_accuracy)
            tbar.set_description(desc)

            
def evaluate(model, criterion, test_dataset):
    
    # set the model to evaluation-mode
    model.eval()
    test_accuracy = 0
    for image_batch, label_batch in zip(*test_dataset):
        # conver numpy to torch.tensor type
        image_batch = torch.Tensor(image_batch) / 255.
        label_batch = torch.LongTensor(label_batch)
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        output = model(image_batch)
        pred = get_prediction(output)
        test_accuracy += get_accuracy(pred, label_batch)
    test_accuracy /= (len(test_dataset[0]))
    return test_accuracy


class TetrisCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 14, kernel_size=12, padding=0)
        self.fc1 = nn.Linear(14, 7)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, img_size, img_size)
        x = F.relu(F.max_pool2d(self.conv1(x), 21))
        x = x.view(batch_size, 14)
        logits = self.fc1(x)
        return logits
    
    
class TetrisCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        ## Fill In Your Code Here ##

        raise NotImplementedError("Erase this line and write down your code.")
        ############################

    def forward(self, x):
        x = x.view(-1, 1, img_size, img_size)
        
        ## Fill In Your Code Here ##

        raise NotImplementedError("Erase this line and write down your code.")
        ############################
        
        return logits
   
