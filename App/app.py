from flask import Flask, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import joblib

app = Flask(__name__)
model = joblib.load('D:\PROJECTS\IMAGE CLASSIFIER\model\model_joblib')

def predict_label(img_path):
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	def extract_features(image):
		model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
		vgg16_features = model.predict(image)
		print(model.summary())
		return vgg16_features

	image = image.reshape(-1, 224, 224, 3)
	features = extract_features(image)
	features = features.reshape(1, -1)
	prediction = model.predict(features)

	return prediction


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	app.run(debug = True)
