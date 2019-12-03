// Typescript
import * as tf from '@tensorflow/tfjs-node';

import word_dict from '../model/1/word_dict.json'

const runModel = async () => {
    try {

        // console.log(word_dict)

        // const model = await tf.loadLayersModel('file://./model/1/model.json');
        // model.predict()
        // console.log(model);
    } catch (error) {
        console.error(error);
    }
}

runModel();
