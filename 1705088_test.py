S = '1705088_train'
m = __import__ (S)
try:
    attrlist = m.__all__
except AttributeError:
    attrlist = dir (m)
for attr in attrlist:
    globals()[attr] = getattr (m, attr)

import os
import numpy as np
import cv2
import pandas as pd

model = m.Model([m.Conv2D(1,6,5,1,2),m.Relu(),m.MaxPooling2D(2,2),m.Conv2D(6,16,5,1,0),m.Relu(),m.MaxPooling2D(2,2),m.FlatteningLayer(),m.FullyConnectedLayer(5*5*16,120),m.Relu(),m.FullyConnectedLayer(120,84),m.Relu(),m.FullyConnectedLayer(84,10),m.Softmax()])
model = m.load_model('1705088_model.pkl')
folder = '../dataset/NumtaDB/testing-d/'



# get the filenames in a folder
def get_filenames(folder):
    filenames = []
    for filename in os.listdir(folder):
        filenames.append(filename)
    return filenames


batch_size = 32
test_shuffle = False

test_dataset = m.DataSet(csv_file='../dataset/NumtaDB/training-d.csv',root_dir='../dataset/NumtaDB/training-d',transform=np.array)
test_data = m.DataLoader(test_dataset,batch_size=batch_size,shuffle=test_shuffle)

print('Test accuracy: ',m.accuracy(test_data,model))
print('Test F1 score: ',m.macro_f1(test_data,model))

np.set_printoptions(suppress = True)
print(m.confusion_matrix(test_data,model))

def image_preporcessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = 255 - image # negavtive
    image = cv2.dilate(image, (2, 2)) # dilate
    mean,var = image.mean(),image.var()
    image = (image-mean)/var

    image = image.reshape(1, 28, 28)
    return image

def predict(image):
    batch_image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    return model.forward(batch_image)[0]

report = {}
report_list = []

for filename in get_filenames(folder):
    if not filename.endswith(('.png','.jpg','.jpeg')):
        continue
    image = cv2.imread(folder+filename)
    image = image_preporcessing(image)
    prediction = predict(image)
    report['filename'] = filename
    report['digit'] = np.argmax(prediction)
    report['probability'] = np.max(prediction)
    report_list.append(report.copy())

df = pd.DataFrame(report_list)
df.to_csv(folder+'1705088_test.csv',index=False)

