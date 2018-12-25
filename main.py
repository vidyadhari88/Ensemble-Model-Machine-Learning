
# coding: utf-8

# In[ ]:




# ## Load MNIST on Python 3.x

# In[1]:


import pickle
import gzip


# In[2]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# ## Load USPS on Python 3.x

# In[3]:


from PIL import Image
import os
import numpy as np


# In[4]:


USPSMat  = []
USPSTar  = []
curPath  = '/Users/vidyach/Desktop/Machine_learning/Project_3/USPSdata/Numerals'

savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            #print(img)
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


# # MNIST Training data processing-seperating the feature values and taret values

# In[5]:


TariningFeatureVector = training_data[0]
TariningFeatureVector = np.transpose(TariningFeatureVector)
print(TariningFeatureVector.shape)
TrainingTargetVector = training_data[1]
print(TrainingTargetVector.shape)


# # MNIST Validation data processing-seperating the feature values and target values

# In[6]:


ValidationFeatureVector = validation_data[0]
ValidationFeatureVector = np.transpose(ValidationFeatureVector)
ValidationTargetVector = validation_data[1]
print(ValidationFeatureVector.shape)
print(ValidationTargetVector.shape)


# # MNIST Testing data processing-seperating the feature values and target values

# In[7]:


TestingFeatureVector = test_data[0]
TestingFeatureVector = np.transpose(TestingFeatureVector)
TestingTargetVector = test_data[1]
print(TestingFeatureVector.shape)
print(TestingTargetVector.shape)


# # USPS Testing data processing

# In[8]:


USPSTestingFeatureVector = np.array(USPSMat)
USPSTestingFeatureVector = np.transpose(USPSTestingFeatureVector)
USPSTestingTargetVector = np.array(USPSTar)
print(USPSTestingFeatureVector.shape)
print(USPSTestingTargetVector.shape)


# # one hot Vector representation of target vector

# In[9]:


def oneHotvector(TargetMatrix):
    oneHotvector = np.zeros((len(TargetMatrix),10))
    #print(len(TargetMatrix))
    for i in range(0,len(TargetMatrix)):
        number = TargetMatrix[i]
        oneHotvector[i,number] = 1
    return oneHotvector
        
        
# one hot vector of training target
oneHotTargetVector = oneHotvector(TrainingTargetVector)
oneHotTargetVector = np.transpose(oneHotTargetVector)
print(oneHotTargetVector.shape)
# one hot vector of validation target
oneHotvalidationTarget = oneHotvector(ValidationTargetVector)
oneHotvalidationTarget = np.transpose(oneHotvalidationTarget)
print(oneHotvalidationTarget.shape)
# one hot vector of testing target
oneHotTestingTarget = oneHotvector(TestingTargetVector)
oneHotTestingTarget = np.transpose(oneHotTestingTarget)
print(oneHotTestingTarget.shape)
# one hot vecotor of USPS target
oneHotUSPSTarget = oneHotvector(USPSTestingTargetVector)
oneHotUSPSTarget = np.transpose(oneHotUSPSTarget)
print(oneHotUSPSTarget.shape)

#Class label matrix - Testing and Training
ClasslabelTarget = np.append(TrainingTargetVector,ValidationTargetVector,axis = 0)
print(ClasslabelTarget.shape)


# # randomly initializing the weigth vector

# In[10]:


weightVector = np.random.randn(TariningFeatureVector.shape[0],10)
print(weightVector.shape)
print(weightVector)


# # pre defined functions

# In[11]:


# function to calculate the predicted value
def GetPredValue(featureMatrix,W):
    Y = 1/ (1 + np.exp(-np.dot(np.transpose(W),featureMatrix)))
    return Y

#function to calculate the accuracy 
def GetAccuracy(valueOut,TargetValue):
    valueOut = np.transpose(valueOut)
    TargetValue = np.transpose(TargetValue)
    right = 0
    wrong = 0
    for i in range(0,valueOut.shape[0]):
        predictValue = np.argmax(valueOut[i])
        #print(TargetValue[i])
        target = np.argmax(TargetValue[i])
        if predictValue == target:
            right += 1
        else:
            wrong +=1
            
    return (right / (right+wrong))*100


# In[12]:


learningRate = 0.02
Regularizer = 0.05

Epoch_size = TariningFeatureVector.shape[0]
Tr_acc_list = []
Val_acc_list = []
Test_acc_list = []
USPS_acc_list = []
USPSTestValues = []
MNISTTestValue = []

for i in range(0,300):
    
    x = 1/ (1 + np.exp(-np.dot(np.transpose(weightVector),TariningFeatureVector)))
    deltaW = -np.dot((oneHotTargetVector - x ),np.transpose(TariningFeatureVector))
    deltaW = np.transpose(deltaW)
    RegDelta = np.dot(Regularizer,weightVector)
    Delta_E = np.add(deltaW,RegDelta)
    Delta_W = -np.dot(learningRate,Delta_E)
    W_next = weightVector + Delta_W
    weightVector = W_next

    #calculating the accuracy of Training data
    valOut = GetPredValue(TariningFeatureVector,W_next)
    TrAccuracy = GetAccuracy(valOut,oneHotTargetVector)
    Tr_acc_list.append(TrAccuracy)
    
    
    #calculating validation accuracy
    valiOut = GetPredValue(ValidationFeatureVector,W_next)
    ValAccuracy = GetAccuracy(valiOut,oneHotvalidationTarget)
    Val_acc_list.append(ValAccuracy)
    
    #calculating testing accuracy
    TestOut = GetPredValue(TestingFeatureVector,W_next)
    MNISTTestValue.append(TestOut)
    TestAccuracy = GetAccuracy(TestOut,oneHotTestingTarget)
    Test_acc_list.append(TestAccuracy)
    
    
    #calculating USPS testing accuracy
    USPSTestOut = GetPredValue(USPSTestingFeatureVector,W_next)
    USPSTestValues.append(USPSTestOut)
    USPSTestAccuracy = GetAccuracy(USPSTestOut,oneHotUSPSTarget)
    USPS_acc_list.append(USPSTestAccuracy)


# # performance results on MNITS Dataset

# In[13]:


import pandas as pd
print("Performance on AMNIST Dataset using :" +str(learningRate)+ "," +str(Regularizer))
print("Training accuracy : " + str(max(Tr_acc_list)))
print("Validation accuracy : " + str(max(Val_acc_list)))
print("Testing accuracy : " + str(max(Test_acc_list)))

df = pd.DataFrame()
df['Training accuracy'] = Tr_acc_list
df.plot(grid=True)

df = pd.DataFrame()
df['Validation accracy'] = Val_acc_list
df.plot(grid=True)

df = pd.DataFrame()
df['AMNIT Testing accuracy'] = Test_acc_list
df.plot(grid=True)



# # Performance results on USPS Dataset

# In[14]:


print("Performance on USPS Dataset using: "+str(learningRate)+ "," +str(Regularizer))
print("Testing accuracy : " + str(max(USPS_acc_list)))

df = pd.DataFrame()
df["USPS Testing Accuracy"] = USPS_acc_list
df.plot(grid=True)


# # confusion matrix USPS

# In[15]:


def ConfusionMatrixError(ConfusionMatrx):
    proba = []
    for i in range(0,len(ConfusionMatrx)):
        sum1 = sum(ConfusionMatrx[i])
        proba.append(np.around(ConfusionMatrx[i,i]/sum1,5))
        #print(sum1)
    return proba


# In[16]:


from sklearn.metrics import confusion_matrix
x = USPS_acc_list.index(max(USPS_acc_list))
PredictedUSPSValues = np.array(USPSTestValues[x])
PredictedUSPSValuesT = np.transpose(PredictedUSPSValues)
PredValueMatrix = np.zeros((PredictedUSPSValuesT.shape[0],1))
for i in range(0,PredictedUSPSValuesT.shape[0]):
    temp = np.argmax(PredictedUSPSValuesT[i])
    PredValueMatrix[i] = temp

PredValueMatrix = PredValueMatrix.astype(np.int)
USPSTestingTargetVector1 = USPSTestingTargetVector.reshape(USPSTestingTargetVector.shape[0],1)
confusionMatrixUSPS = confusion_matrix(USPSTestingTargetVector1,PredValueMatrix)
probabilityUSPSLog = ConfusionMatrixError(confusionMatrixUSPS)
print(confusionMatrixUSPS)
print(probabilityUSPSLog)


# # confusion matrix MNITS

# In[17]:


y = Test_acc_list.index(max(Test_acc_list))
predictedValuesMNITS = np.array(MNISTTestValue[y])
predictedValuesMNITST = np.transpose(predictedValuesMNITS)
predValuesMNITS = np.zeros((predictedValuesMNITST.shape[0],1))
for i in range(0,predictedValuesMNITST.shape[0]):
    temp = np.argmax(predictedValuesMNITST[i])
    predValuesMNITS[i] = temp
    
predValuesMNITS = predValuesMNITS.astype(np.int)
TestingTargetVector1 = TestingTargetVector.reshape(TestingTargetVector.shape[0],1)
confusionMatrixMNITS = confusion_matrix(TestingTargetVector1,predValuesMNITS)
probabilityMNITSLog = ConfusionMatrixError(confusionMatrixMNITS)
print(confusionMatrixMNITS)
print(probabilityMNITSLog)


# # Neural networks implementation

# In[18]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping

inputSize = TariningFeatureVector.shape[0]
fristLayerHiddenUnits = 400
seclayerHU = 100
outputSize = 10

def init_model():
    model = Sequential()
    model.add(Dense(fristLayerHiddenUnits,input_dim = inputSize,use_bias = True))
    model.add(Activation("relu"))
    
    model.add(Dense(seclayerHU,use_bias = True))
    model.add(Activation("relu"))
    
    model.add(Dense(outputSize))
    model.add(Activation("softmax"))
    
    
    model.summary()
    
    model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    
    return model


# In[19]:


model = init_model()

# combining training and validation data
concatFeatureVector = np.append(TariningFeatureVector,ValidationFeatureVector,axis=1)
concatFeatureVector = np.transpose(concatFeatureVector)

targetVector = np.append(oneHotTargetVector,oneHotvalidationTarget,axis=1)
targetVector = np.transpose(targetVector)


# In[20]:


validationData = 0.167
num_epochs = 100
model_batch_size = 100
tb_batch_size = 32
early_patience = 100

earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

history = model.fit(concatFeatureVector,targetVector,validation_split=validationData,
                epochs=num_epochs,batch_size=model_batch_size
                , callbacks = [earlystopping_cb])


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # testing on MNIST

# In[22]:


TestingFeatureTranspose = np.transpose(TestingFeatureVector)
oneHotTestTarget = np.transpose(oneHotTestingTarget)
predictedTestLabel = np.zeros(oneHotTestTarget.shape)
for i in range(0,TestingFeatureTranspose.shape[0]):
    x = model.predict(TestingFeatureTranspose[i].reshape(-1,784))
    predictedTestLabel[i] = x
predictedTestLabel = np.transpose(predictedTestLabel)
oneHotTranspose = np.transpose(oneHotTestTarget)
TestAccuracy = GetAccuracy(predictedTestLabel,oneHotTranspose)


# In[23]:


print("Testing accuracy of MNIST Dataset : " +str(TestAccuracy))


# # confusion matrix MNITS

# In[24]:


predictedTestLabelT = np.transpose(predictedTestLabel)
predValuesMNITSNN = np.zeros((predictedTestLabelT.shape[0],1))
for i in range(0,predictedTestLabelT.shape[0]):
    predValuesMNITSNN[i] = np.argmax(predictedTestLabelT[i]) 
predValuesMNITSNN = predValuesMNITSNN.astype(np.int)
confusionMatrix = confusion_matrix(TestingTargetVector1,predValuesMNITSNN)
probMNITSNN = ConfusionMatrixError(confusionMatrix) 
print(confusionMatrix)
print(probMNITSNN)


# # Testing On USPS

# In[25]:


USPSTestingFeatureVectorT = np.transpose(USPSTestingFeatureVector)
#print(USPSTestingFeatureVectorT.shape)
oneHotUSPSTarget = np.transpose(oneHotUSPSTarget)
preditedValue = np.zeros(oneHotUSPSTarget.shape)

for i in range(0,USPSTestingFeatureVectorT.shape[0]):
    x = model.predict(USPSTestingFeatureVectorT[i].reshape(-1,784))
    preditedValue[i] = x
preditedValue = np.transpose(preditedValue)
oneHotUSPSTarget = np.transpose(oneHotUSPSTarget)
USPSAccr = GetAccuracy(preditedValue,oneHotUSPSTarget)
print("USPS Test accuracy : "+str(USPSAccr))


# # Confusion matrix USPS

# In[26]:


preditedValueT = np.transpose(preditedValue)
predValuesUSPS = np.zeros((preditedValueT.shape[0],1))
for i in range(0,preditedValueT.shape[0]):
    predValuesUSPS[i] = np.argmax(preditedValueT[i])
    
confusionMat = confusion_matrix(USPSTestingTargetVector1,predValuesUSPS)
probaUSPSNN = ConfusionMatrixError(confusionMat)
print(confusionMat)
print(probaUSPSNN)


# # convolusion neural network

# In[27]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

outputSize = 10

def init_cnnmodel():
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(outputSize, activation='softmax'))

    model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    
    return model


# In[28]:


cnnModel = init_cnnmodel()


# In[29]:


validationData = 0.167
num_epochs = 15
model_batch_size = 1000
tb_batch_size = 32
early_patience = 100

x_train = concatFeatureVector.reshape(60000,28,28,1)

earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
print(concatFeatureVector.shape)
history = cnnModel.fit(x_train,targetVector,validation_split=validationData,
                epochs=num_epochs,batch_size=model_batch_size
                , callbacks = [earlystopping_cb])


# In[30]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # Testing on MNITS dataset

# In[44]:


TestingFeatureTransposeT = np.transpose(TestingFeatureVector)
oneHotTestTarget = np.transpose(oneHotTestingTarget)
predictedTestLabelTCNN = np.zeros(oneHotTestTarget.shape)
predictedTestLabelTCNN = cnnModel.predict(TestingFeatureTransposeT.reshape(10000,28,28,1))
predictedTestLabelTCNN = np.transpose(predictedTestLabelTCNN)
oneHotTranspose = np.transpose(oneHotTestTarget)
TestAccuracy = GetAccuracy(predictedTestLabelTCNN,oneHotTranspose)
print("CNN MNITS Test Accuracy:" +str(TestAccuracy))


# # Testing on USPS dataset

# In[32]:


USPSTestingFeatureVectorT = np.transpose(USPSTestingFeatureVector)
oneHotUSPSTargetTr = np.transpose(oneHotUSPSTarget)
predictedTestLabelTCNNUSPS = np.zeros(oneHotUSPSTargetTr.shape)
predictedTestLabelTCNNUSPS = cnnModel.predict(USPSTestingFeatureVectorT.reshape(19999,28,28,1)) 
predictedTestLabelTCNNUSPS = np.transpose(predictedTestLabelTCNNUSPS)
oneHotUSPSTargetT = np.transpose(oneHotUSPSTargetTr)
USPSAccr = GetAccuracy(predictedTestLabelTCNNUSPS,oneHotUSPSTargetT)
print("USPS Test accuracy : "+str(USPSAccr))


# # Confusion matrix of MNIST

# In[33]:


from sklearn.metrics import confusion_matrix

predictedTestLabelTCNNT = np.transpose(predictedTestLabelTCNN)
predValuesMNITSCNN = np.zeros((predictedTestLabelTCNNT.shape[0],1))
for i in range(0,predictedTestLabelTCNNT.shape[0]):
    predValuesMNITSCNN[i] = np.argmax(predictedTestLabelTCNNT[i]) 
predValuesMNITSCNN = predValuesMNITSCNN.astype(np.int)
#print(predValuesMNITSCNN.shape)
TestingTargetVector1 = TestingTargetVector.reshape(TestingTargetVector.shape[0],1)
confusionMatrix = confusion_matrix(TestingTargetVector1,predValuesMNITSCNN)
probMNITSCNN = ConfusionMatrixError(confusionMatrix) 
print("confusion matrix CNN: MNITS")
print(confusionMatrix)
print(probMNITSCNN)


predictedTestLabelTCNNUSPST = np.transpose(predictedTestLabelTCNNUSPS)
predValuesUSPSCNN = np.zeros((predictedTestLabelTCNNUSPST.shape[0],1))
for i in range(0,predictedTestLabelTCNNUSPST.shape[0]):
    predValuesUSPSCNN[i] = np.argmax(predictedTestLabelTCNNUSPST[i])
USPSTestingTargetVector1 = USPSTestingTargetVector.reshape(USPSTestingTargetVector.shape[0],1)
confusionMat = confusion_matrix(USPSTestingTargetVector1,predValuesUSPSCNN)
probaUSPSCNN = ConfusionMatrixError(confusionMat)
print("confusion matrix CNN: USPS")
print(confusionMat)
print(probaUSPSCNN)


# # SVM Approach

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel='rbf');
classifier.fit(concatFeatureVector,ClasslabelTarget)


# # testing on MNIST dataset

# In[45]:


predictedValues = classifier.predict(TestingFeatureTranspose)


# In[36]:


def accr(predValues,TargetValues):
    right = 0
    wrong = 0
    for i in range(0,len(predValues)):
        if predValues[i] == TargetValues[i]:
            right += 1
        else:
            wrong += 1
    return ((right/(right+wrong))*100)

Accuracy  = accr(predictedValues,TestingTargetVector)
print("MNITS Testing Accuracy : " + str(Accuracy))


# # Testing on USPS dataset

# In[37]:


USPSPredValues = classifier.predict(USPSTestingFeatureVectorT)
USPSAccr = accr(USPSPredValues,USPSTestingTargetVector)
print("USPS Testing Accuracy : "+str(USPSAccr))


# # Confusion Matrix

# In[38]:


#MNITS
confusionMatrixMNITS = confusion_matrix(TestingTargetVector,predictedValues)
#usps
confusionMatrixUSPS = confusion_matrix(USPSTestingTargetVector,USPSPredValues)
print("confusion matrix MNITS:")
print(confusionMatrixMNITS)
probMNITSCNNx = ConfusionMatrixError(confusionMatrixMNITS)
print(probMNITSCNNx)
print("confusion matrix USPS:")
print(confusionMatrixUSPS)
probMNITSCNNy = ConfusionMatrixError(confusionMatrixUSPS)
print(probMNITSCNNy)


# # Random Forest Approach

# In[39]:


from sklearn.ensemble import RandomForestClassifier
claasifier2 = RandomForestClassifier(n_estimators=500)
claasifier2.fit(concatFeatureVector,ClasslabelTarget)


# In[40]:


# Testing on MNIST
TestingFeatureTranspose = np.transpose(TestingFeatureVector)
print(TestingFeatureTranspose.shape)
predValue = claasifier2.predict(TestingFeatureTranspose)


# In[41]:


Accr = accr(predValue,TestingTargetVector)
print("MNIST Testing Accr: "+str(Accr))


# In[42]:


# Testing on USPS
USPSTestingFeatureVectorT = np.transpose(USPSTestingFeatureVector)
USPSPredValues1 = claasifier2.predict(USPSTestingFeatureVectorT)
USPSAccr1 = accr(USPSPredValues1,USPSTestingTargetVector)
print("USPS Testing Accuracy : "+str(USPSAccr1))


# # confusion matrix

# In[43]:


from sklearn.metrics import confusion_matrix
#MNITS
confusionMatrxMNITS = confusion_matrix(TestingTargetVector,predValue)
ProbaMNITSRF = ConfusionMatrixError(confusionMatrxMNITS)
#USPS
confusionMatrxUSPS = confusion_matrix(USPSTestingTargetVector,USPSPredValues1)
ProbaUSPSSRF = ConfusionMatrixError(confusionMatrxUSPS)

print("Confusion matrix MNITS:")
print(confusionMatrxMNITS)
print(ProbaMNITSRF)
print("confusion matrix USPS:")
print(confusionMatrxUSPS)
print(ProbaUSPSSRF)


# # ensemble the model using hard voting

# In[124]:


def ensembleMat(predValuesMNITSLogit,predValuesMNITSNN,predValuesMNITSCNN,predValuesMNITSSVM,predValuesMNITSRF):
    ensemblePred = np.zeros(predValuesMNITSLogit.shape)
    for i in range(0,predValuesMNITSLogit.shape[0]):
        temp = predValuesMNITSLogit[i],predValuesMNITSNN[i],predValuesMNITSCNN[i],predValuesMNITSSVM[i],predValuesMNITSRF[i]
        count = 0
        output = 20
        for number in range(0,10):
            tempCount = temp.count(number)
            if tempCount > count:
                count = tempCount
                output = number
        ensemblePred[i] = output
    return ensemblePred


# # Accuracy on MNITS

# In[128]:


predValuesMNITSLogit = predValuesMNITS
predValuesMNITSNN = predValuesMNITSNN
predValuesMNITSCNNx = predValuesMNITSCNN
predValuesMNITSSVM = predictedValues.reshape(predictedValues.shape[0],1)
predValuesMNITSRF = predValue.reshape(predValue.shape[0],1)

ensemblePredMNITS = ensembleMat(predValuesMNITSLogit,predValuesMNITSNN,predValuesMNITSCNNx,predValuesMNITSSVM,predValuesMNITSRF)


TestingTargetVectorT = TestingTargetVector.reshape(TestingTargetVector.shape[0],1)
#print(TestingTargetVectorT.shape)
#print(ensemblePredMNITS.shape)

print("Emsemble model accr on MNITS: " +str(accr(ensemblePredMNITS,TestingTargetVectorT)))


# # Accuracy on USPS

# In[130]:


PredValueMatrixLogit = PredValueMatrix
PredValueMatrixNN = predValuesUSPS
PredValueMatrixCNN = predValuesUSPSCNN
PredValueMatrixSVM = USPSPredValues.reshape(USPSPredValues.shape[0],1)
PredValueMatrixRF = USPSPredValues1.reshape(USPSPredValues1.shape[0],1)
ensemblePredUSPS = ensembleMat(PredValueMatrixLogit,PredValueMatrixNN,PredValueMatrixCNN,PredValueMatrixSVM,PredValueMatrixRF)
USPSTestingTargetVectorR = USPSTestingTargetVector.reshape(USPSTestingTargetVector.shape[0],1)
#print(ensemblePredUSPS.shape)
#print(USPSTestingTargetVectorR.shape)

print("Ensemble mdel accr on USPS: "+str(accr(ensemblePredUSPS,USPSTestingTargetVectorR)))


# # confusion matrix

# In[138]:


confusionMatrixMNIST = confusion_matrix(TestingTargetVectorT,ensemblePredMNITS)
confusionMatrixUSPS = confusion_matrix(USPSTestingTargetVectorR,ensemblePredUSPS)
print("Confusion matrix of MNITS: ")
print(confusionMatrixMNIST)
probax = ConfusionMatrixError(confusionMatrixMNIST)
print(probax)
print("confusion matrix of USPS: ")

print(confusionMatrixUSPS)
probay = ConfusionMatrixError(confusionMatrixUSPS)
print(probay)

