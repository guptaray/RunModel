#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import pickle
# Load the saved model using pickle
with open('KNN_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('Random_Forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('SVM_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('LR_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()


# In[2]:


def make_prediction(model, item):
    input_data = pd.DataFrame([[
        item.sepal_length,
        item.sepal_width,
        item.petal_length,
        item.petal_width
    ]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(input_data)
    print(item)
    print('Predicted Value:', prediction.tolist()[0])
    return prediction.tolist()[0]


# In[3]:


@app.post("/predict_knn")
async def predict_knn(item: Item):
    print('KNN')
    return {"prediction": make_prediction(knn_model, item)}


# In[4]:


@app.post("/predict_rf")
async def predict_rf(item: Item):
    print('RF')
    return {"prediction": make_prediction(rf_model, item)}


# In[5]:


@app.post("/predict_svm")
async def predict_svm(item: Item):
    print('SVM')
    return {"prediction": make_prediction(svm_model, item)}


# In[5]:


@app.post("/predict_lr")
async def predict_lr(item: Item):
    print('LR')
    return {"prediction": make_prediction(lr_model, item)}

