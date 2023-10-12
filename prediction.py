#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa 
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
import soundfile
 


# In[2]:


pip install librosa


# In[3]:


paths = []
labels = []
for dirname, _, filenames in os.walk(r"C:\Users\khush\Downloads\archive\TESS Toronto emotional speech set data"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')


# In[4]:


len(paths)


# In[5]:


labels[:5]


# In[6]:


paths[:5]


# In[7]:


df=pd.DataFrame()
df['speech']=paths
df['label']=labels


# In[8]:


df.head()


# In[9]:


df['label'].value_counts()


# In[10]:


sns.countplot(df['label'])


# In[11]:


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    


# In[12]:


def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    


# In[14]:


emotion='fear'
path=np.array(df['speech'][df['label']==emotion])[1]
data,sampling_rate= librosa.load(path)
waveplot(data,sampling_rate,emotion)
spectogram(data,sampling_rate,emotion)
Audio(path)


# In[ ]:


emotion = 'angry'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[ ]:


emotion = 'disgust'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[ ]:


emotion = 'neutral'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[ ]:


emotion = 'sad'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[ ]:


emotion = 'ps'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[ ]:


emotion = 'happy'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[ ]:


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    return mfcc


# In[ ]:


extract_mfcc(df['speech'][0])


# In[ ]:


X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))


# In[ ]:


X_mfcc.shape


# In[ ]:


X = [x for x in X_mfcc]
X = np.array(X)
X.shape


# In[ ]:


plt.figure(figsize=(25,10))
librosa.display.specshow(X,x_axis="time")
plt.colorbar(format="%+2f")
plt.show()


# In[ ]:


X_mfcc


# In[ ]:


X = np.expand_dims(X, -1)
X.shape


# In[ ]:


print(df['label'])


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=df.drop(['label'],axis=1)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['speech']])


# In[ ]:


y = y.toarray()
y.shape


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


playsound(r'C:\\Users\\khush\\Downloads\\archive\\TESS Toronto emotional speech set data\\OAF_neutral\\OAF_kill_neutral.wav')


# In[ ]:


model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


emotions={
  '01':'neutral',
  '02':'ps',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fear',
  '07':'disgust',
  }
#These are the emotions User wants to observe more :
observed_emotions=['ps', 'happy','neutral','angry','sad','fear', 'disgust']


# In[ ]:





# In[ ]:


print(observed_emotions)


# In[ ]:


feature=X
x.append(feature)
y.append([emotion,file_name])


# In[ ]:


pip install glob


# In[ ]:


def load_data(test_size=0.33):
    x,y=[],[]
    answer = 0
    for file in (r'C:\Users\khush\Downloads\archive\TESS Toronto emotional speech set data'):
        file_name=os.path.basename(file)
        feature=extract_mfcc(file_name)
        x.append(feature)
        y.append([emotion,file_name])
    return train_test_split(np.array(x), y, test_size=test_size, random_state=7)


# In[ ]:


pip install playsound


# In[ ]:


from playsound import playsound


# In[ ]:


playsound(file_name)


# In[ ]:


x_train,x_test,y_trai,y_tes=train_test_split(x,y,test_size=0.25)
print(np.shape(x_train),np.shape(x_test), np.shape(y_trai),np.shape(y_tes))
y_test_map = np.array(y_tes).T
y_test = y_test_map[0]
test_filename = y_test_map[1]
y_train_map = np.array(y_trai).T
y_train = y_train_map[0]
train_filename = y_train_map[1]
print(np.shape(y_train),np.shape(y_test))
print(*test_filename,sep="\n")


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])


# In[ ]:


y = y.toarray()
y.shape


# In[ ]:


x_train,x_test,y_train,y_test=load_data(test_size=0.25)
print(np.shape(x_train),np.shape(x_test), np.shape(y_trai),np.shape(y_tes))
y_test_map = np.array(y_tes).T
y_test = y_test_map[0]
test_filename = y_test_map[1]
y_train_map = np.array(y_trai).T
y_train = y_train_map[0]
train_filename = y_train_map[1]
print(np.shape(y_train),np.shape(y_test))
print(*test_filename,sep="\n")


# In[ ]:


print((x_train[0], x_test[0]))


# In[ ]:


pip install keras


# In[ ]:


pip install tensorflow


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(13,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X, y, validation_split=0.2, epochs=20, batch_size=64)


# In[ ]:


epochs = list(range(20))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[ ]:


pip install sounddevice


# In[ ]:





# In[ ]:


pip install wavio


# In[ ]:


pip install scipy


# In[ ]:


import sounddevice
from scipy.io.wavfile import write
import wavio as wv


# In[ ]:


freq=44100
duration= 3


# In[ ]:


recording=sounddevice.rec((duration*freq),samplerate=freq,channels=2)


# In[ ]:


sounddevice.wait()


# In[ ]:


write("sample.wav",freq,recording)


# In[ ]:


wv.write("sample.wav", recording, freq, sampwidth=2)


# In[ ]:


pip install playsound


# In[ ]:


from playsound import playsound


# In[ ]:


playsound(r'C:\Users\khush\sample.wav')


# In[ ]:


file = r'C:\Users\khush\sample.wav'
# data , sr = librosa.load(file)
# data = np.array(data)
ans =[]
new_feature = extract_mfcc(file)
ans.append(new_feature)
ans = np.array(ans)
# data.shape

emotion=model.perdict(ans)


# In[ ]:




