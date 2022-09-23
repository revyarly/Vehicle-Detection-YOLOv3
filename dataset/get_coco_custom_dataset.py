import json
import requests
def get_training_dataset():
    count = 0
    file_path_train = '/home/sh1v/Mini_Project/Yolo_CoCo_Dataset/custom_train2017_dataset_coco/'
    with open('annotations/instances_train2017_custom.json') as json_file:
        data = json.load(json_file)
        print(len(data['images']))
        for i in range(len(data['images'])):
            str = data['images'][i]['coco_url'].split('/')
            url = data['images'][i]['coco_url']
            r = requests.get(url, allow_redirects=True)
            open(file_path_train+str[-1], 'wb').write(r.content)
            file1 = open("gettrainvalno2017.txt","a")
            file1.write(file_path_train+str[-1]+'\n')

def get_validation_dataset():
    count = 0
    file_path_val = '/home/sh1v/Mini_Project/Yolo_CoCo_Dataset/custom_val2017_dataset_coco/'
    with open('annotations/instances_val2017_custom.json') as json_file:
        data = json.load(json_file)
        print("Number of images: ",len(data['images']))
        for i in range(len(data['images'])):
            str = data['images'][i]['coco_url'].split('/')
            url = data['images'][i]['coco_url']
            r = requests.get(url, allow_redirects=True)
            open(file_path_val+str[-1],'wb').write(r.content)
            file2 = open("getvalidationno2017.txt","a")
            file2.write(file_path_val+str[-1]+'\n')
            count += 1
            print(count)
get_validation_dataset()
