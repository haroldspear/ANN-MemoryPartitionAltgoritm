from keras.layers import Input, Dense, Flatten
from keras.models import Model, load_model, model_from_json
from keras.layers import concatenate
import numpy as np
import pandas as pd
import os

CURVES_SIZE=3470 #Minimun size of curves
model="model" #Define global model

def train_evaluate_predict_split(dataset_dir, train_percent=.7, evaluate_percent=.2, seed=None):
    dataset=os.listdir(dataset_dir)
    #Get random part of dataset and split
    for csv in dataset:
        df=pd.read_csv(dataset_dir+"/"+csv, header=None)
        np.random.seed(seed)
        permutation = np.random.permutation(df.index)
        total = len(df.index)
        train_end = int(train_percent * total)
        evaluate_end = int(evaluate_percent * total) + train_end
        train = df.ix[permutation[:train_end]]
        evaluate = df.ix[permutation[train_end:evaluate_end]]
        predict = df.ix[permutation[evaluate_end:]]
        train.to_csv("train/"+csv, encoding='utf-8', index=False, header=None)
        evaluate.to_csv("evaluate/"+csv, encoding='utf-8', index=False, header=None)
        predict.to_csv("predict/"+csv, encoding='utf-8', index=False, header=None)

def get_dataset(dataset_dir):
    dataset=os.listdir(dataset_dir)
    curves_names=[]
    daraset_Y=[]
    flag_Y=True
    size=0
    #Get curves and labels(outputs) from dataset
    for csv in dataset:
        filename=dataset_dir+"/"+csv
        curves_number=int(filename.split(".")[0].split("_")[1])
        increment_1=5*(curves_number-2)
        increment_2=4*(curves_number-2)
        dataset_csv = np.genfromtxt(filename, delimiter=",", usecols=np.arange(curves_number,15+increment_1))
        size=size+dataset_csv.shape[0]
        print("Current file: ", filename)
        if flag_Y:
            dataset_Y = dataset_csv[:,9+increment_2:13+increment_2]
            flag_Y=False
        else:
            dataset_Y=np.append(dataset_Y,dataset_csv[:,9+increment_2:13+increment_2], axis=0)
        df=pd.read_csv(filename, header=None)
        curves_names=curves_names+df.ix[:,0:1*(curves_number-1)].values.tolist()
    curves_x=[]
    curves_y=[]
    curve_x_1=[]
    curve_y_1=[]
    flag1=True
    size_xy=size
    #Get samples from all curves in a row for all dataset (inputs)
    for row in curves_names:
        c_num=len(row)
        flag=True
        for curve in row:
            curve=np.genfromtxt("traces/"+curve.split(".")[0]+".csv", delimiter=",")
            cx=curve[:CURVES_SIZE,0].reshape(CURVES_SIZE,1)
            cy=curve[:CURVES_SIZE,1].reshape(CURVES_SIZE,1)
            if flag:
                curve_x_1=cx
                curve_y_1=cy
                flag=False
            else:
                curve_x_1=np.concatenate((curve_x_1,cx), axis=0)
                curve_y_1=np.concatenate((curve_y_1,cy), axis=0)
        if c_num > 2:
            id = np.random.choice(np.arange(CURVES_SIZE*c_num), CURVES_SIZE*2, replace=False)
            id = np.sort(id)
            curve_x_1=curve_x_1[id]
            curve_y_1=curve_y_1[id]
        if flag1:
            curves_x=curve_x_1
            curves_y=curve_y_1
        else:
            curves_x=np.concatenate((curves_x,curve_x_1), axis=0)
            curves_y=np.concatenate((curves_y,curve_x_1), axis=0)
        flag1=False
    X=[]
    curves_x=curves_x.reshape(size_xy,1,int(curves_x.shape[0]/size_xy))
    curves_y=curves_y.reshape(size_xy,1,int(curves_y.shape[0]/size_xy))
    X.append(curves_x)
    X.append(curves_y)
    Y=dataset_Y[:,[1,3]]
    #Define winner algorithm, best utility value,
    Y=np.where(Y[:,0]>Y[:,1], 1, 0)
    Y=Y.reshape(size,1,1)
    unique, counts = np.unique(Y, return_counts=True)

    return X, Y

#get_dataset("train/comb_3.csv")

def def_model():
    global model
    #Build the model
    #Input layer
    input_curves_x=Input(shape=(1,CURVES_SIZE*2), name="i_cx")
    input_curves_y=Input(shape=(1,CURVES_SIZE*2), name="i_cy")
    #Define dense layer, each neuron in this layer will be fully connected
    #to all neurons in the next layer
    x_0x_1 = Dense(128, activation="relu")(input_curves_x)
    x_0x_1 = Dense(64, activation="relu")(x_0x_1)
    x_0x_1 = Dense(32, activation="relu")(x_0x_1)
    x_0x_1 = Dense(16, activation="relu")(x_0x_1)
    x_0x_1 = Dense(8, activation="relu")(x_0x_1)
    x_0x_1 = Dense(4, activation="relu")(x_0x_1)
    x_0x_1 = Dense(1, activation="relu")(x_0x_1)
    x_0x_1 = Model(inputs=input_curves_x, outputs=x_0x_1)

    x_0y_1 = Dense(128, activation="relu")(input_curves_y)
    x_0y_1 = Dense(64, activation="relu")(x_0y_1)
    x_0y_1 = Dense(32, activation="relu")(x_0y_1)
    x_0y_1 = Dense(16, activation="relu")(x_0y_1)
    x_0y_1 = Dense(8, activation="relu")(x_0y_1)
    x_0y_1 = Dense(4, activation="relu")(x_0y_1)
    x_0y_1 = Dense(1, activation="relu")(x_0y_1)
    x_0y_1 = Model(inputs=input_curves_y, outputs=x_0y_1)

    combined = concatenate([x_0x_1.output, x_0y_1.output])
    z = Dense(4, activation="relu")(combined)
    z = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=[x_0x_1.input, x_0y_1.input], outputs=z)
    compile_model()

def load_model(model_name="model_curves"):
    global model
    #Load model from disk
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(model_name+".h5")
    compile_model()
    print("Model loaded")

def compile_model():
    global model

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

def predict():
    global model
    X_test, Y_test=get_dataset("predict")
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    #print('First prediction:', predictions[0])


def train_and_evaluate(dataset_dir="train", epochs=100, batch_size=2000, model_name="model_curves"):
    global model
    X_train, Y_train=get_dataset(dataset_dir)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    print("End Train...")
    X_eval, Y_eval=get_dataset("evaluate")
    scores = model.evaluate(X_eval, Y_eval)

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model_json = model.to_json()
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_name+".h5")
    print("Saved model to disk")
