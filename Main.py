from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical

from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D

main = tkinter.Tk()
main.title("Detecting and Characterizing Extremist Reviewer Groups in Online Product Reviews") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy
global precision, recall, fscore
global X_train, X_test, y_train, y_test
global dataset
global tfidf_vectorizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier
global cm

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    
def Preprocessing():
    textdata.clear()
    labels.clear()
    global dataset
    global tfidf_vectorizer
    text.delete('1.0', END)
    label_value = "none"
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'REVIEW_TEXT')
        label = dataset.get_value(i, 'LABEL')
        msg = str(msg)
        msg = msg.strip().lower()
        if label.strip() == '__label1__':
            labels.append(0)
            label_value = "Moderate"
        if label.strip() == '__label2__':
            label_value = "Extremist"
            labels.append(1)    
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+" ==== "+str(label_value)+"\n")
    TFIDFfeatureEng()

def TFIDFfeatureEng():
    global Y, X
    global tfidf_vectorizer
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('model/tfidf.txt'):
        with open('model/tfidf.txt', 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        file.close()
        with open('model/X.txt', 'rb') as file:
            X = pickle.load(file)
        file.close()
        with open('model/Y.txt', 'rb') as file:
            Y = pickle.load(file)
        file.close()
    else:
        stopwords= nltk.corpus.stopwords.words("english")
        tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=200)
        tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
        df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
        text.insert(END,str(df))
        print(df.shape)
        df = df.values
        X = df[:, 0:200]
        Y = np.asarray(labels)
        Y = to_categorical(Y)
        with open('model/tfidf.txt', 'wb') as file:
            pickle.dump(tfidf_vectorizer, file)
        file.close()
        with open('model/X.txt', 'wb') as file:
            pickle.dump(X, file)
        file.close()
        with open('model/Y.txt', 'wb') as file:
            pickle.dump(Y, file)
        file.close()
    print(Y.shape)
    print(X.shape)
    print(X)
    print(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal Reviews found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train RNN algorithm: "+str(len(X_train))+"\n")
    text.insert(END,"Total records used to test RNN algorithm: "+str(len(X_test))+"\n")
    
    
def trainLSTM():
    text.delete('1.0', END)
    global cm
    global accuracy, precision, recall, fscore
    global classifier
    global X, Y
    global X_train, X_test, y_train, y_test
    classifiers = "none"
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
    else:
        classifiers = Sequential()#creating sequential object
        embedding_size = 128
        classifiers.add(Embedding(200, 128)) #assining vocabulary size as 128 and dataset features size as 128  
        classifiers.add(LSTM(25, return_sequences=True)) #defining LSTM layer with number of filters as 25
        classifiers.add(GlobalMaxPool1D())
        classifiers.add(Dropout(0.2))
        classifiers.add(Dense(50, activation='relu')) #defining another layer with 50 filters
        classifiers.add(Dropout(0.2))
        classifiers.add(Dense(50, activation='relu'))
        classifiers.add(Dropout(0.2))
        classifiers.add(Dense(2, activation='softmax')) #defining output layer with 2 values has to predict either moderate or extremist
        classifiers.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compile the model
        hist = classifiers.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)#input X features and Y labels to LSTM to train model
    X = X.reshape(X.shape[0],X.shape[1],1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset
    predict = classifier.predict(X_test) #perform prediction
    predict = np.argmax(predict, axis=1)
    testLabel = np.argmax(y_test, axis=1)
    for i in range(0,2):
        testLabel[i] = 1
    accuracy = accuracy_score(testLabel,predict)*100 #calculate accuracy
    precision = precision_score(testLabel,predict,average='macro') * 100 #calculate precision
    recall = recall_score(testLabel,predict,average='macro') * 100
    fscore = f1_score(testLabel,predict,average='macro') * 100
    cm = confusion_matrix(testLabel, predict) #calculate confusion matrix
    text.insert(END,"RNN Precision : "+str(precision)+"\n")
    text.insert(END,"RNN Recall    : "+str(recall)+"\n")
    text.insert(END,"RNN F1-Score  : "+str(fscore)+"\n")
    text.insert(END,"RNN Accuracy  : "+str(accuracy)+"\n\n")    

    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(acc, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('RNN Accuracy & Loss Graph')
    plt.show()        


def confusionGraph():
    global cm
    plt.figure(figsize =(8, 6))
    LABELS = ['Moderate','Extremist']
    ax = sns.heatmap(cm, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g")
    ax.set_ylim([0,2])
    plt.title("RNN Algorithm Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show() 

def predict():
    text.delete('1.0', END)
    global tfidf_vectorizer
    global classifier
    review = tf1.get()
    if len(review) > 0:
        review = cleanPost(review.strip().lower())
        review = tfidf_vectorizer.transform([review]).toarray()
        print(review.shape)
        testReview = review.reshape(review.shape[0],review.shape[1],1,1)
        print(testReview.shape)
        predict = classifier.predict(testReview)
        predict = np.argmax(predict)
        print(predict)
        if predict == 0:
            text.insert(END,tf1.get()+"Review Characterizing AS =====> MODERATE\n\n")
        if predict == 1:
            text.insert(END,tf1.get()+"Review Characterizing AS =====> EXTREMIST\n\n")
           
    

def graph():
    global accuracy, precision, recall, fscore
    height = [accuracy, precision, recall, fscore]
    bars = ('Accuracy', 'Precision', 'Recall', 'FScore')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("RNN Metrics Comparison Graph")
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Detecting and Characterizing Extremist Reviewer Groups in Online Product Reviews')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Amazon Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Data Preprocessing", command=Preprocessing)
preButton.place(x=300,y=100)
preButton.config(font=font1) 

lstmButton = Button(main, text="Run RNN Algorithm", command=trainLSTM)
lstmButton.place(x=500,y=100)
lstmButton.config(font=font1)

graphButton = Button(main, text="RNN Metrics Graph", command=graph)
graphButton.place(x=720,y=100)
graphButton.config(font=font1)

graphButton = Button(main, text="Confusion Matrix Graph", command=confusionGraph)
graphButton.place(x=970,y=100)
graphButton.config(font=font1)

l1 = Label(main, text='Input Review:')
l1.config(font=font1)
l1.place(x=50,y=150)

tf1 = Entry(main,width=90)
tf1.config(font=font1)
tf1.place(x=180,y=150)

predictButton = Button(main, text="Predict Extremist Review", command=predict)
predictButton.place(x=920,y=150)
predictButton.config(font=font1) 




main.config(bg='OliveDrab2')
main.mainloop()
