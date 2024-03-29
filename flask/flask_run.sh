#!/bin/bash

source ../venv_keras/bin/activate
# python -V

uwsgi --master --http 0.0.0.0:5000 --wsgi-file keras-fine-food-server.py \
      --callable app --processes 1 --threads 1 --virtualenv ../venv_keras \
      --py-autoreload=3 
      # --logto serverlog.log

# uwsgi --master --http 0.0.0.0:5000 --wsgi-file keras-fine-food-server.py \
#       --callable app --processes 2 --threads 2 --virtualenv venv_keras --logto serverlog.log &


# python -c "import tensorflow as tf; from keras.models import load_model; print(tf.__version__)"