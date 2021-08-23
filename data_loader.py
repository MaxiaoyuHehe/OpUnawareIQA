import torch
import torchvision
import folders


class DataLoader02(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, img_indx, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        if (dataset == 'kadid') or (dataset == 'cuz') or (dataset == 'cuzG') or (dataset == 'cuzP') or (
                dataset == 'cuzE') or (dataset == 'cuzEG') or (dataset == 'cuzEP') or (dataset == 'tid'):
            # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
            # Test transforms
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])

        if dataset == 'kadid':
            self.data = folders.KadidFolder(
                root='H:\kadis700k\kadis700k', index=img_indx, transform=transforms)
        elif dataset == 'tid':
            self.data = folders.TID2013Folder(
                root='D:\ImageDatabase\\tid2013', index=img_indx, transform=transforms)
        elif dataset == 'cuz':
            self.data = folders.CuzFolder(
                root='E:\CUZ2021', index=img_indx, transform=transforms)
        elif dataset == 'cuzG':
            self.data = folders.CuzGFolder(
                root='E:\CUZ2021', index=img_indx, transform=transforms)
        elif dataset == 'cuzP':
            self.data = folders.CuzPFolder(
                root='E:\CUZ2021', index=img_indx, transform=transforms)
        elif dataset == 'cuzE':
            self.data = folders.CuzEFolder(
                root='D:\ImageDatabase\\tid2013', index=img_indx, transform=transforms)
        elif dataset == 'cuzEG':
            self.data = folders.CuzEGFolder(
                root='D:\ImageDatabase\\tid2013', index=img_indx, transform=transforms)
        elif dataset == 'cuzEP':
            self.data = folders.CuzEPFolder(
                root='D:\ImageDatabase\\tid2013', index=img_indx, transform=transforms)


    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader
