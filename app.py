from keras.models import load_model
from flask import Flask
from flask_cors import CORS
import Utils
model = load_model('my_model.h5')

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def hello():
    return {'hello': 'Thế Giới'}


@app.route('/classify/<string:text>')
def classify(text):
    res = Utils.predict_file(model, str(text))
    return res


if __name__ == "__main__":
    app.run(debug=True)
