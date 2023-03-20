import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import net
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

class Emnist():

    def __init__(self):
        
        batch_size = 4

        trainset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=ToTensor())
        testset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=ToTensor())

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        self.classes = tuple(trainset.classes)

        # # Show sample of data from training set
        # dataiter = iter(self.trainloader)
        # images, labels = next(dataiter)
        # print(images.shape)
        # self.imshow(torchvision.utils.make_grid(images, nrow=8))
        # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

        self.net = net.Net()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.train(self.trainloader, self.net, loss_fn, optimizer)
        self.test(self.testloader, self.net, loss_fn)

        PATH = './emnist_net.pth'
        torch.save(self.net.state_dict(), PATH)


    
    def train(self, dataloader, model, loss_fn, optimizer):
        """ 
        Training loop for neural network

        :param dataloader: input dataset
        :param model: model being trained
        :param loss_fn: loss function
        :param optimizer: optimizer
        """
        for epoch in range(1): # loop over dataset multiple times
            running_loss = 0.0
            for batch, data in enumerate(dataloader, 1):
                # get inputs; data is a list of [input, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, backward, optimize
                outputs = self.net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # stats
                running_loss += loss.item()
                if batch % 2000 == 1999:
                    print(f'[{epoch+1}, {batch+1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

    def test(self, dataloader, model, loss_fn):
        """
        Tests accuracy of the trained model on testing dataset
        Also tests accuracy for each class
        """
        correct = 0
        total = 0
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        print(f'Accuracy: {100*correct // total} %')

        for classname, correct_count in correct_pred.items():
            if total_pred[classname] != 0:
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


                

    def imshow(self, img): # show images
        img = img
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (2, 1, 0)))
        plt.show()


    

        

if __name__ == '__main__':
    e = Emnist()