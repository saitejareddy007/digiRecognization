import numpy as np
import matplotlib.pyplot as pt
import matplotlib.image as ptimg
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("train.csv").as_matrix()
clf=DecisionTreeClassifier()

#training dataset
xtrain=data[0:21000,1:]
train_label=data[0:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest=data[21000:,1:]
actual_label=data[21000:,0]

d=xtest[6]
d.shape=(28,28)

pt.imshow(255-d,cmap='gray')
print(clf.predict([xtest[6]]))
pt.show()


#im=ptimg.imread('image.png')
# show the image
#pt.imshow(im,cmap='gray')
#print(im)
#pt.show()