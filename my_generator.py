#encode:utf-8
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt  
import torchvision

import os 
import random
import numpy as np 

from utils import split_integer,show_image


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


def get_default_transform():
    return transforms.Compose([Rotate(random.choice([0,90,180,270]))])


class Omniglot_generator(Dataset):
    def __init__(self,data_folder, n_classes, n_samples, n_test = 0, transform = get_default_transform()):
        self.n_test = n_test
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.folder_names = []
        for alpha_folder in os.listdir(data_folder):
            for character_folder in os.listdir(data_folder + '/' + alpha_folder):
                
                self.folder_names.append(data_folder+'/'+alpha_folder + '/' + character_folder + '/')

        self.transform = transform



    def __getitem__(self, idx):

        images = []
        if self.n_test == 0:
            test_split = []
        else:
            test_split = split_integer(self.n_test,self.n_classes)

        #train images
        train_dirs = []
        for class_names in random.sample(self.folder_names,self.n_classes):
            train_dirs.append(class_names)
            filenames_list = os.listdir(class_names)
            for filename in random.sample(filenames_list, self.n_samples):
                img = Image.open(class_names + filename)
                #img = self.transform(img)
                img = img.resize((28,28), resample=Image.LANCZOS)
                images.append(np.array(img, dtype = np.double))

        #test images
        if self.n_test != 0:
            for i,class_names in enumerate(train_dirs):
                filenames_list = os.listdir(class_names)
                for filename in random.sample(filenames_list, test_split[i]):
                    img = Image.open(class_names + filename)
                    #img = self.transform(img)
                    img = img.resize((28,28), resample=Image.LANCZOS)
                    images.append(np.array(img, dtype = np.double))

        
        #train labels
        labels = []
        for i in range(self.n_classes):
            for j in range(self.n_samples):
                labels.append(i)
        #test labels
        if self.n_test != 0:
            for i in range(self.n_classes):
                for j in range(test_split[i]):
                    labels.append(i) 



        data = []
        for i in range(self.n_classes*self.n_samples):
            data.append((images[i],labels[i]))
        random.shuffle(data)
        for i in range(self.n_classes*self.n_samples):
            images[i],labels[i] = data[i]
        n_train = self.n_classes*self.n_samples
        data = []
        if self.n_test != 0:
            for i in range(self.n_test):
                data.append((images[n_train+i],labels[n_train+i]))
            random.shuffle(data)
            for i in range(self.n_test):
                images[n_train+i],labels[n_train+i] = data[i]

        images = np.asarray(images)
        labels = np.asarray(labels)  
        return images, labels


    def __len__(self):
        return 100000

from utils import show_image

def visulize(tensor):
    imgs = torchvision.utils.make_grid(tensor*0.5+0.5).cpu()
    plt.imshow(imgs.permute(1,2,0).detach().numpy())
    plt.show()


def unit_test():
    
    dataset = Omniglot_generator('./omniglot/images_background',2,8,8)
    dataloader = DataLoader(dataset,batch_size = 1, shuffle = False)
    train_iter = iter(dataloader)
    images,labels = next(train_iter)
    images = torch.transpose(images,1,0)
    print(labels)
    visulize(images)
    


if __name__ == '__main__':
    unit_test()

        



