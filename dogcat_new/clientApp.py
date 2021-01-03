from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from com_in_ineuron_ai_utils.utils import decodeImage
from predict import dogcat

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = dogcat(self.filename)


@app.route("/", methods=['GET'])  # when we are passing something through url
@cross_origin()  # that is done by get method
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])  # when we are passing something through body of code
@cross_origin()  # that is done by post method
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictiondogcat()
    return jsonify(result)


# port = int(os.getenv("PORT"))  # comment this code if you are testing in local
if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=8000, debug=True)
