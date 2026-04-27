import os
import random
import torch
import torchvision
from torchvision.datasets.folder import default_loader

class CustomImageFolderWithNegativeSample(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolderWithNegativeSample, self).__init__(root, transform=transform)
        self.imgs = self.samples

    def __getitem__(self, index):
        # positive
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # negative data
        all_classes = list(range(len(self.classes)))
        all_classes.remove(target)  # remove positive
        negative_class = random.choice(all_classes)  # choose a negative class
        negative_indices = [i for i, (_, class_idx) in enumerate(self.imgs) if class_idx == negative_class]  # all the classes
        negative_index = random.choice(negative_indices)  # choose a negative sample
        negative_path, negative_target = self.imgs[negative_index]
        negative_img = self.loader(negative_path)
        if self.transform is not None:
            negative_img = self.transform(negative_img)

        return img, target, negative_img, negative_target
    


# # 使用示例
# imagenet_root = 'path_to_imagenet'
# IN_aug_type = torchvision.transforms.Compose([
#     # 在这里定义你的数据增强
# ])

# train_dataset = CustomImageFolderWithNegativeSample(
#     os.path.join(imagenet_root, 'train'),
#     transform=IN_aug_type
# )
