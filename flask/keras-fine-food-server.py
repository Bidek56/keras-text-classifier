import flask
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle, os

# App config.
DEBUG = True
app = flask.Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def hello():
    return flask.render_template('index.html')


# Perform these actions when the first
# request is made
@app.before_first_request
def load_model_to_app():

    source_path = '../model/1/'

    # Load the model
    app.loaded_model = load_model(os.path.join(source_path, 'Food_Reviews.h5'))

    with open(os.path.join(source_path, 'tokenizer.pickle'), 'rb') as handle:
        app.loaded_tokenizer = pickle.load(handle)
    
#     # Save the graph to the app framework.
#     # app.graph = tf.get_default_graph()


@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        # print(flask.request.form)

        text = flask.request.form['name']
        # print(f'Text: {text}')

        if text:
            max_len=250
            seq = app.loaded_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=max_len)
            pred = app.loaded_model.predict_classes(padded)
            flask.flash(f'Text to be rated:{text}')
            flask.flash(f'Predicted score:{pred}')

        else:
            flask.flash( "Error: Text cannot be empty")

    return flask.redirect('/')

if __name__ == "__main__":
    app.run()
