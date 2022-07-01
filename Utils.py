import re
import pandas as pd
from pyvi import ViTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df_res = pd.read_csv('DataSet.csv', encoding='utf-8')

datacmt = []
for d in df_res['review_text']:
    e = ViTokenizer.tokenize(str(d))
    datacmt.append(e)
labelcmt = df_res['lable']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(datacmt)


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


def remove_URL(text):
    text = str(text)
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def predict_file(model, tweet):
    test_word = str(tweet)
    test_word = test_word.lower()
    test_word = test_word.replace('[^\w\s]', '')
    test_word = test_word.replace('[^0-9a-zA-Z]', '')
    test_word = remove_URL(test_word)
    test_word = remove_emoji(test_word)
    tw = tokenizer.texts_to_sequences([test_word])
    print(tw)
    tw = pad_sequences(tw, 20)
    prediction = int(model.predict(tw).round().item())
    print(model.predict(tw))
    if (prediction == 0):
        return ('POSITIVE')
    else:
        return ('NEGATIVE')
