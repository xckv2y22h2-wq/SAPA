import json 
datapath = './datasets/ImageNet/imagenet_class_index.json'
with open(datapath, 'r') as f:
    class_idx = json.load(f)

class_names = [class_idx[str(i)][1] for i in range(1000)]
print (class_names[:10])  # Print first 10 class names to verify