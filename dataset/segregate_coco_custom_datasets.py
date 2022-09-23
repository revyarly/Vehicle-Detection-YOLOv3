import pycocotools
import gluoncv
import mxnet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm_notebook as tqdm
train_dataset = gluoncv.data.COCODetection('.',splits=['instances_train2017'])
val_dataset = gluoncv.data.COCODetection('.',splits=['instances_val2017'])
names= pd.read_csv("./coco.names")
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))
#Loading random image and it's respective label
train_image, train_label = train_dataset[1234]
#Getting bounding boxes of objects present in the loaded image.
bounding_boxes = train_label[:, :4]
# Getting Classes of objects present in the image.
class_ids = train_label[:, 4:5]
print(class_ids)
#Visualizing the image with bounding boxes around the recognized objects
gluoncv.utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
 labels=class_ids, class_names=train_dataset.classes)
plt.savefig("example4.png")
Id_counts = {}
for k in range(79):
    Id_counts[names.values[k][0]]=0
for i in tqdm(range(len(train_dataset))):
    train_image, train_label = train_dataset[i]
    bounding_boxes = train_label[:, :4]
    class_ids = train_label[:, 4:5]
    for j in range(79):
        if j in class_ids:
            Id_counts[names.values[j][0]]+=1
print(Id_counts)
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.bar(range(80), Id_counts.values(), width= 0.8,color='g')
plt.xticks(range(80),Id_counts.keys() , rotation=90)
plt.show()