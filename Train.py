import pandas as pd
import re
import numpy as np
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_short, strip_numeric
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from pyvi import ViTokenizer, ViPosTagger

df_res = pd.read_csv('DataSet.csv', encoding='utf-8')


def remove_URL(text):
    text = str(text)
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_number(docs):
    return [strip_numeric(doc) for doc in docs]


def replace_multiple_whitespaces(docs):
    return [strip_multiple_whitespaces(doc) for doc in docs]


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_one_letter_word(docs):
    return [strip_short(doc) for doc in docs]

datacmt=[]
for d in df_res['review_text']:
    e=ViTokenizer.tokenize(str(d))
    datacmt.append(e)
labelcmt=df_res['lable']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(datacmt)
datacmtbow = tokenizer.texts_to_sequences(datacmt)
datacmtbow= pad_sequences(datacmtbow, maxlen=20)

vocab_size = len(datacmtbow) + 1
encoded_docs = tokenizer.texts_to_sequences(df_res.review_text)
padded_sequence = pad_sequences(encoded_docs, maxlen=20)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length,
                                     input_length=20) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(120,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(40))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',
                           metrics=['accuracy'])

history = model.fit(padded_sequence, labelcmt,
                    validation_split=0.2, epochs=3, batch_size=8)

# save the model to disk
model.save('my_model.h5')
