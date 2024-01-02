from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO
import numpy as np

app = Flask(__name__)

model = load_model('my_cnn_model.h5')  # Load your saved model

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        # Get the uploaded image
        image_file = request.files['image']

        # Load image data into BytesIO
        image_data = image_file.read()
        image_bytes = BytesIO(image_data)

        # Process the image
        test_image = image.load_img(image_bytes, target_size=(64, 64))  # Load from BytesIO
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Image Pre Processing
        # ... 

        # Make prediction
        prediction = model.predict(test_image)
        prediction_class = 'Cat' if prediction[0][0] >= 0.5 else 'Dog'

        return render_template('index.html', prediction=prediction_class)

    return render_template('index.html')



if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)  # Run on localhost
    app.run(host='0.0.0.0', port=8080)  # Run on localhost
