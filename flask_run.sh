#!/bin/bash

# source ~/keras-env-3.7/bin/activate
# python -V

export FLASK_APP=keras-fine-food-server.py
export FLASK_ENV=development
export FLASK_DEBUG=1
# flask run

uwsgi --master --http 0.0.0.0:5000 --wsgi-file keras-fine-food-server.py \
      --callable app --processes 1 --threads 1 --virtualenv venv_keras \
      --py-autoreload=3

# uwsgi --master --http 0.0.0.0:5000 --wsgi-file keras-fine-food-server.py \
#       --callable app --processes 2 --threads 2 --virtualenv venv_keras --logto serverlog.log &


# python -c "import tensorflow as tf; from keras.models import load_model; print(tf.__version__)"