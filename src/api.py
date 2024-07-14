from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('models/model_pneumonia_detection.h5')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = image.load_img(img, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    prediction_class = 'Pneumonia' if prediction > 0.5 else 'Normal'
    confidence = float(prediction[0][0])
    
    return jsonify({
        'prediction': prediction_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
