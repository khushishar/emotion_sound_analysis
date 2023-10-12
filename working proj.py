#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import librosa
import numpy as np


# In[2]:


from keras.models import load_model
model = load_model("emorecog.h5")


# In[3]:


model.summary()


# In[5]:


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# In[6]:


import sounddevice
from scipy.io.wavfile import write
import wavio as wv


# In[7]:


freq=44100
duration= 3


# In[14]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])


# In[8]:


recording=sounddevice.rec((duration*freq),samplerate=freq,channels=2)
sounddevice.wait()
write(r"C:/Users/khush/Downloads/electron-GUI-EmotionA/electron-GUI-EmotionA/enginesample.wav",freq,recording)
wv.write(r"C:/Users/khush/Downloads/electron-GUI-EmotionA/electron-GUI-EmotionA/enginesample.wav", recording, freq, sampwidth=2)


# In[13]:


file = r"C:/Users/khush/Downloads/electron-GUI-EmotionA/electron-GUI-EmotionA/enginesample.wav"
ans =[]
new_feature = extract_mfcc(file)
ans.append(new_feature)
ans = np.array(ans)
y_pred=model.predict(ans)
emo=enc.inverse_transform(y_pred)
print(emo)


# In[ ]:




