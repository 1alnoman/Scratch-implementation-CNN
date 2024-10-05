import numpy as np
import pandas as pd
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

np.random.seed(789)

class MeanSquaredLoss:
    def score(self,y,y_hat):
        self.y=y
        self.y_hat=y_hat
        return np.mean((y-y_hat)**2)
    def gradient(self,y,y_hat):
        return 2*(self.y_hat-self.y)/self.y.size

class cross_entropy:
    def score(self,y,y_hat):
        self.y=y
        self.y_hat=y_hat
        return -np.mean(np.sum(y*np.log(y_hat),axis=1))
    def gradient(self,y,y_hat):
        return -self.y/self.y_hat/self.y.size

# calculate training accuracy
def accuracy(data_loader,model):
    correct=0
    total=0
    for x,y in data_loader:
        y_pred=model.forward(x)
        y_pred=np.argmax(y_pred,axis=1)
        correct+=(y_pred==y).sum()
        total+=y.shape[0]
    return correct/total

# macro-f1 score
def macro_f1(data_loader,model):
    y_pred=[]
    y_true=[]
    for x,y in data_loader:
        y_pred.append(np.argmax(model.forward(x),axis=1))
        y_true.append(y)
    y_pred=np.concatenate(y_pred)
    y_true=np.concatenate(y_true)
    f1=[]
    for i in range(10):
        tp=np.sum((y_pred==i)&(y_true==i))
        fp=np.sum((y_pred==i)&(y_true!=i))
        fn=np.sum((y_pred!=i)&(y_true==i))
        f1.append(2*tp/(2*tp+fp+fn))
    return np.mean(f1)

# print confusion matrix
def confusion_matrix(data_loader,model):
    y_pred=[]
    y_true=[]
    for x,y in data_loader:
        y_pred.append(np.argmax(model.forward(x),axis=1))
        y_true.append(y)
    y_pred=np.concatenate(y_pred)
    y_true=np.concatenate(y_true)
    confusion=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            confusion[i,j]=np.sum((y_pred==i)&(y_true==j))
    return confusion

class DataSet:
    def __init__(self,csv_file,root_dir,transform=None):
        self.data=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        img_name=self.data.loc[idx,['filename']].values[0]
        img_path=os.path.join(self.root_dir,img_name)
        image=cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = 255 - image # negavtive
        image = cv2.dilate(image, (2, 2)) # dilate
        # image = image.astype('float32')/255.
        mean,var = image.mean(),image.var()
        image = (image-mean)/var

        image = image.reshape(1, 28, 28)
        
        
        label=self.data.loc[idx,['digit']].values[0]
        if self.transform:
            image=self.transform(image)
        return image,label

class DataLoader:
    def __init__(self,dataset,batch_size=1,shuffle=False):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.index=0
        self.length=len(dataset)
        self.indices=np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indices)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index>=self.length:
            self.index=0
            raise StopIteration
        batch_indices=self.indices[self.index:self.index+self.batch_size]
        self.index+=self.batch_size
        batch_x=[]
        batch_y=[]
        for i in batch_indices:
            x,y=self.dataset[i]
            batch_x.append(x)
            batch_y.append(y)
        return np.array(batch_x),np.array(batch_y)

# one hot encoding of y
def one_hot(y):
    y_hot=np.zeros((y.size,10))
    y_hot[np.arange(y.size),y]=1
    return y_hot

# xaiver initialization
def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return np.random.uniform(low=low,high=high,size=(fan_in,fan_out))

# make pickle of a model
def save_model(model,file_name):
    with open(file_name,'wb') as f:
        pickle.dump(model,f)

# load pickle of a model
def load_model(file_name):
    with open(file_name,'rb') as f:
        model=pickle.load(f)
    return model

def getWindow(input,out_shape,kernel_size,stride,padding,dilate):
    if dilate!=0:
        input = np.insert(input,range(1,input.shape[2]),0,axis=2)
        input = np.insert(input,range(1,input.shape[3]),0,axis=3)

    input = np.pad(input,((0,0),(0,0),(padding,padding),(padding,padding)),'constant')
    
    batch_size = input.shape[0]
    input_channels = input.shape[1]
    output_height = out_shape[2]
    output_width = out_shape[3]
    strided_shape = (batch_size,input_channels,output_height,output_width,kernel_size,kernel_size)
    strided_stride = (input.strides[0],input.strides[1],input.strides[2]*stride,input.strides[3]*stride,input.strides[2],input.strides[3])
    strided = np.lib.stride_tricks.as_strided(input,shape=strided_shape,strides=strided_stride)

    return strided
# fully connected layer with mini-batch gradient descent
class FullyConnectedLayer:
    def __init__(self,input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.weights=xavier_init(input_size,output_size)
        self.bias=np.zeros(output_size)
    def forward(self,x):
        self.x=x
        self.z=np.dot(x,self.weights)+self.bias
        return self.z
    def backward(self,delta):
        self.d_weights=np.dot(self.x.T,delta)
        self.d_bias=np.sum(delta,axis=0)
        self.delta=np.dot(delta,self.weights.T)
        return self.delta
    def update(self,learning_rate,batch_size):
        self.weights-=(learning_rate/batch_size)*self.d_weights
        self.bias-=(learning_rate/batch_size)*self.d_bias

class FlatteningLayer:
    def __init__(self):
        pass
    def forward(self,x):
        self.x=x
        self.batch_size=x.shape[0]
        self.input_channels=x.shape[1]
        self.input_height=x.shape[2]
        self.input_width=x.shape[3]
        self.output_size=self.input_channels*self.input_height*self.input_width
        self.z=x.reshape(self.batch_size,self.output_size)
        return self.z
    def backward(self,delta):
        self.delta=delta.reshape(self.batch_size,self.input_channels,self.input_height,self.input_width)
        return self.delta
    def update(self,learning_rate,batch_size):
        pass

# Relu activation function
class Relu:
    def forward(self,x):
        self.x=x
        return np.maximum(0,x)
    def backward(self,delta):
        self.delta=delta.copy()
        self.delta[self.x<=0]=0
        return self.delta
    def update(self,learning_rate,batch_size):
        pass

# softmax activation function
class Softmax:
    def forward(self,x):
        self.x=x
        self.exp=np.exp(x)
        self.sum=np.sum(self.exp,axis=1,keepdims=True)
        return self.exp/self.sum
    def backward(self,delta):
        self.delta=delta.copy()
        return self.delta
    def update(self,learning_rate,batch_size):
        pass

class Conv2D:
    def __init__(self,input_channels,output_channels,kernel_size,stride,padding):
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.weights=xavier_init(output_channels,input_channels*kernel_size*kernel_size)
        self.weights=self.weights.reshape(output_channels,input_channels,kernel_size,kernel_size)
        self.bias=np.zeros(output_channels)
    def forward(self,x):
        self.x = x
        out_shape = (x.shape[0],self.output_channels,int((x.shape[2]-self.kernel_size+2*self.padding)/self.stride+1),int((x.shape[3]-self.kernel_size+2*self.padding)/self.stride+1))
        self.strided_x = getWindow(self.x,out_shape,self.kernel_size,self.stride,self.padding,0)
        self.z = np.einsum('bihwkl,oikl->bohw',self.strided_x,self.weights)
        self.z = self.z + self.bias.reshape(1,self.bias.shape[0],1,1)
        return self.z
    def backward(self,delta):
        padding = self.kernel_size-1 if self.padding==0 else self.padding
        dilation = self.stride - 1
        strided_delta = getWindow(delta,self.x.shape,self.kernel_size,1,padding,dilation)
        rotated_weights = np.rot90(self.weights,2,(2,3))

        self.delta_x = np.einsum('bohwkl,oikl->bihw',strided_delta,rotated_weights)
        self.delta_weights = np.einsum('bihwkl,bohw->oikl',self.strided_x,delta)
        self.delta_bias = np.sum(delta,axis=(0,2,3))
        return self.delta_x
    def update(self,learning_rate,batch_size):
        self.weights -= learning_rate*self.delta_weights/batch_size
        self.bias -= learning_rate*self.delta_bias/batch_size

class MaxPooling2D:
    def __init__(self,kernel_size,stride):
        self.kernel_size=kernel_size
        self.stride=stride

    def forward(self,x):
        self.x = x
        out_shape = (x.shape[0],x.shape[1],int((x.shape[2]-self.kernel_size)/self.stride+1),int((x.shape[3]-self.kernel_size)/self.stride+1))
        self.strided_x = getWindow(self.x,out_shape,self.kernel_size,self.stride,0,0)
        self.z = np.max(self.strided_x,axis=(4,5))
        return self.z
    def backward(self,delta):
        if self.kernel_size == self.stride:
            self.delta_x = np.repeat(np.repeat(delta,self.kernel_size,axis=2),self.kernel_size,axis=3)
            self.mask = np.equal(self.x,self.z.repeat(self.kernel_size,axis=2).repeat(self.kernel_size,axis=3))
            self.delta_x = self.delta_x*self.mask
        else:
            self.delta_x = None
        return self.delta_x
    def update(self,learning_rate,batch_size):
        pass
class Model:
    def __init__(self,layers):
        self.layers=layers
    def forward(self,x):
        for layer in self.layers:
            x=layer.forward(x)
        return x
    def backward(self,delta):
        for layer in reversed(self.layers):
            delta=layer.backward(delta)
        return delta
    def update(self,learning_rate,batch_size):
        for layer in self.layers:
            layer.update(learning_rate,batch_size)
    def fit_batch(self,criteria,x,y,learning_rate=0.01):
        batch_size = x.shape[0]
        y_pred = self.forward(x)
        for i in range(batch_size):
            y_pred_i = y_pred[i]
            y_i = y[i]
            delta = criteria.gradient(y_pred_i,y_i)

            self.backward(delta)
            self.update(learning_rate,batch_size)


if __name__ == '__main__':
    model = Model([Conv2D(1,6,5,1,2),Relu(),MaxPooling2D(2,2),Conv2D(6,16,5,1,0),Relu(),MaxPooling2D(2,2),FlatteningLayer(),FullyConnectedLayer(5*5*16,120),Relu(),FullyConnectedLayer(120,84),Relu(),FullyConnectedLayer(84,10),Softmax()])
    # model = load_model('lenetmodel.pkl')

    criteria = cross_entropy()


    n_epochs = 20
    learning_rates = [5,1,0.1,0.01]
    batch_size = 32
    train_shuffle = True
    test_shuffle = True

    train_dataset_a = DataSet(csv_file='../dataset/NumtaDB/training-a.csv',root_dir='../dataset/NumtaDB/training-a',transform=np.array)
    train_data_a = DataLoader(train_dataset_a,batch_size=batch_size,shuffle=train_shuffle)

    train_dataset_b = DataSet(csv_file='../dataset/NumtaDB/training-b.csv',root_dir='../dataset/NumtaDB/training-b',transform=np.array)
    train_data_b = DataLoader(train_dataset_b,batch_size=batch_size,shuffle=train_shuffle)

    train_dataset_c = DataSet(csv_file='../dataset/NumtaDB/training-c.csv',root_dir='../dataset/NumtaDB/training-c',transform=np.array)
    train_data_c = DataLoader(train_dataset_c,batch_size=batch_size,shuffle=train_shuffle)

    train_data_list = [train_data_a,train_data_c]
    validation_data_list = [train_data_b]

    # report_df = pd.DataFrame(columns=['epoch','learning_rate','epoch','train_loss','train_accuracy','train_macro_f1','validation_loss','validation_accuracy','validation_macro_f1'])

    report_list = []
    for learning_rate in learning_rates:
        model = Model([Conv2D(1,6,5,1,2),Relu(),MaxPooling2D(2,2),Conv2D(6,16,5,1,0),Relu(),MaxPooling2D(2,2),FlatteningLayer(),FullyConnectedLayer(5*5*16,120),Relu(),FullyConnectedLayer(120,84),Relu(),FullyConnectedLayer(84,10),Softmax()])
        print('learning_rate: ',learning_rate)
        for epoch in range(n_epochs):
            print('epoch: ',epoch)

            report = {}
            report['learning_rate'] = learning_rate
            report['epoch'] = epoch

            train_loss = 0
            train_accuracy = 0
            train_macro_f1 = 0
            validation_loss = 0
            validation_accuracy = 0
            validation_macro_f1 = 0
            for train_data in reversed(train_data_list):
                for i_batch,(x,y) in tqdm(enumerate(train_data)):
                    y_true=one_hot(y)

                    y_pred=model.forward(x)
                    loss=criteria.score(y_true,y_pred)
                    delta=2*(y_pred-y_true)/y_true.size
    
                    model.backward(delta)
                    model.update(learning_rate,batch_size)    

                train_loss += loss
                train_accuracy += accuracy(train_data,model)
                train_macro_f1 += macro_f1(train_data,model)
            report['train_loss'] = train_loss/len(train_data_list)
            report['train_accuracy'] = train_accuracy/len(train_data_list)
            report['train_macro_f1'] = train_macro_f1/len(train_data_list)
            for validation_data in validation_data_list:
                for i_batch,(x,y) in tqdm(enumerate(validation_data)):
                    y_true = one_hot(y)
                    y_pred = model.forward(x)
                    loss = criteria.score(y_true,y_pred)

                validation_loss += loss
                validation_accuracy += accuracy(validation_data,model)
                validation_macro_f1 += macro_f1(validation_data,model)
            report['validation_loss'] = validation_loss/len(validation_data_list)
            report['validation_accuracy'] = validation_accuracy/len(validation_data_list)
            report['validation_macro_f1'] = validation_macro_f1/len(validation_data_list)

            print(report)
            report_list.append(report)

        save_model(model,f'lenetmodel_lr_{learning_rate}.pkl')
    
    report_df = pd.DataFrame(report_list)
    report_df.to_csv('report.csv')