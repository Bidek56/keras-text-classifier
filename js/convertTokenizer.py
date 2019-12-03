import tensorflow as tf
import pickle
import json

# tokenizer = tf.keras.preprocessing.text.Tokenizer()
model_version='1'

'''
with open('../tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

with open( f'model/{model_version}/word_dict.json' , 'w' ) as file:
	json.dump( loaded_tokenizer.word_index , file )
'''

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
model = tf.keras.models.load_model('../Food_Reviews.h5')
export_path = './model/1'

model.save(export_path, save_format='tf')

# # Fetch the Keras session and save the model
# # The signature definition is defined by the input and output tensors
# # And stored with the default serving key
# with tf.keras.backend.get_session() as sess:
#     tf.saved_model.simple_save(
#         sess,
#         export_path,
#         inputs={'input_image': model.input},
#         outputs={t.name:t for t in model.outputs})


# converter = tf.lite.TFLiteConverter.from_saved_model( '../Food_Reviews.h5' )
# converter.post_training_quantize = True
# tflite_buffer = converter.convert()
# open( f'model/{model_version}/model.tflite' , 'wb' ).write( tflite_buffer )