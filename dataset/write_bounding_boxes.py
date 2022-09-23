import json
import sys
def training_dataset_annotations():
    with open('annotations/instances_train2017_custom.json') as json_file:
        data = json.load(json_file)
        for i in range(len(data['images'])):
            x1 = y1 = w = h = x2 = y2 = x_centroid = y_centroid = x_center = y_center = width_coco = height_coco = width = height = image_id = 0
            writable_string = ""
            height = data['images'][i]['height']
            width = data['images'][i]['width']
            image_id = data['images'][i]['id']
            for annotation in data['annotations']:
                if(image_id == annotation['image_id']):
                    temp = str(image_id).zfill(12)
                    x1 = annotation['bbox'][0]
                    y1 = annotation['bbox'][1]
                    w = annotation['bbox'][2]
                    h = annotation['bbox'][3]
                    x2 = x1 + w
                    y2 = y1 + h
                    x_centroid = (x1+ x2)/2
                    y_centroid = (y1 +y2)/2
                    object_class_id = annotation['category_id']
                    x_center = x_centroid / width
                    y_center = y_centroid / height
                    width_coco = w / width
                    height_coco = h / height
                    if(x_center > 1 and y_center > 1):
                        sys.exit("Error in reading file")
                    writable_string = str(object_class_id)+" "+str(x_center)+" "+str(y_center)+" "+str(width_coco)+" "+str(height_coco)+"\n"
                    # print("x1: ",x1," bbox_width: ",w," bbox_height: ",h," x2: ",x2," y1: ",y1," y2: ",y2," x_centroid: ",x_centroid," y_centroid: ",y_centroid," x_center: ",x_center," y_center: ",y_center)
                    bounding_box_file = open("/home/sh1v/Mini_Project/Yolo_CoCo_Dataset/labels/train/"+temp+".txt", "a+")
                    bounding_box_file.write(writable_string)

def validation_dataset_annotations():
    with open('annotations/instances_val2017_custom.json') as json_file:
        data = json.load(json_file)
        for i in range(500,len(data['images'])):
            height = data['images'][i]['height']
            width = data['images'][i]['width']
            image_id = data['images'][i]['id']
            print("Image ID in images object: ",image_id)
            for annotation in data['annotations']:
                if(image_id == annotation['image_id']):
                    temp = str(image_id).zfill(12)
                    x1 = annotation['bbox'][0]
                    y1 = annotation['bbox'][1]
                    w = annotation['bbox'][2]
                    h = annotation['bbox'][3]
                    x2 = x1 + w
                    y2 = y1 + h
                    x_centroid = (x1+ x2)/2
                    y_centroid = (y1 +y2)/2
                    object_class_id = annotation['category_id']
                    x_center = x_centroid / width
                    y_center = y_centroid / height
                    width_coco = w / width
                    height_coco = h / height
                    writable_string = str(object_class_id)+" "+str(x_center)+" "+str(y_center)+" "+str(width_coco)+" "+str(height_coco)+"\n"
                    print("x1: ",x1," bbox_width: ",w," bbox_height: ",h," x2: ",x2," y1: ",y1," y2: ",y2," x_centroid: ",x_centroid," y_centroid: ",y_centroid," x_center: ",x_center," y_center: ",y_center)
                    print(writable_string)
                    bounding_box_file = open("/home/sh1v/Mini_Project/Yolo_CoCo_Dataset/labels/val/"+temp+".txt", "a+")
                    bounding_box_file.write(writable_string)
            x1 = y1 = w = h = x2 = y2 = x_centroid = y_centroid = x_center = y_center = width_coco = height_coco = width = height = image_id = 0
            writable_string = ""


validation_dataset_annotations()