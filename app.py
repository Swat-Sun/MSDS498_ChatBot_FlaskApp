"""Main Flask App for ChatBot."""
from fileinput import filename
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from img_etl import ImageCaptionPredict, make_prediction
from medbot import predict
from PIL import Image
import requests
from io import BytesIO
import json

# Define upload folder path:
UPLOAD_FOLDER = os.path.join("static",'uploads')
RESPONSE_FOLDER = os.path.join("response")
# Define allowed files:
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESPONSE_FOLDER'] = RESPONSE_FOLDER

@app.route("/")
def welcome():
    """Chatbot API Home Page."""
    return render_template("home.html")

@app.route("/generate_text", methods=["GET", "POST"])
def generate_text():
    """User input text."""
    return render_template("gen_text.html")

@app.route("/gpt2chatbot", methods=["GET", "POST"])
def gpt2chatbot():
    """User input text."""
    return render_template("gpt2_healthbot.html")

@app.route("/api/text_echo", methods = ['POST'])
def text_echo():
    """Echo user input text from REST POST."""
    return jsonify(request.json)

@app.route("/api/image", methods = ["POST"])
def recieve_img():
    return jsonify(request.json)

@app.route("/send_text", methods = ["POST"])
def text_output():
    """Send Text output as JSON."""
    if request.method == "POST":
        msg = str(request.form["usr_input"])
        return jsonify({'data':msg})

@app.route("/text_api")
def text_api():
    """Demo Text API"""
    data = "Hello World!"
    return jsonify({'data':data})

@app.route("/output_img_rest", methods = ['POST'])
def img_api():
    """Demo Image API"""
    if request.method == "POST":
        img = request.files["img"]
        img_filename = secure_filename(img.filename)
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))

        return send_from_directory("static/uploads", img_filename)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload image."""
    return render_template("upload.html")

@app.route("/upload_img_rest", methods=["GET", "POST"])
def upload_img_rest():
    """Upload image for REST API."""
    return render_template("upload_img_rest.html")

@app.route("/output", methods=["POST"])
def output():
    """Return output of image submission."""
    if request.method == "POST":
        # Get uploaded files
        f1 = request.files["img1"]
        f2 = request.files["img2"]

        # Extract uploaded data files
        img1_filename = secure_filename(f1.filename)
        img2_filename = secure_filename(f2.filename)

        # Upload file:
        f1.save(os.path.join(app.config['UPLOAD_FOLDER'], img1_filename))
        f2.save(os.path.join(app.config['UPLOAD_FOLDER'], img2_filename))

        # Create file path:
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_filename)
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_filename)

        prediction = make_prediction(img1_path, img2_path)

        return render_template(
            "output.html",
            image_1=img1_path,
            image_2=img2_path,
            prediction = prediction,
        )

@app.route("/api/uploadImage", methods=["POST"])
def uploadImage():
    """Return output of image submission."""
    if request.method == "POST":
        image_path = request.form.get('image_path')

        # img = data_uri_to_cv2_img(image_path)

        # model,tokenizer=create_model()
        predicted_caption=ImageCaptionPredict(image_path)

        return predicted_caption


@app.route("/api/chatbot_img_prediction", methods = ["GET","POST"])
def chatbot_img_prediction():
    if request.method == "POST":
        # Get uploaded files
        r = request.json
        #Get image names:
        global img1_name
        global img2_name
        img1_name = r["name1"]
        img2_name = r["name2"]
        #Get image locations:
        img1_url = r["contentUrl1"]
        img2_url = r["contentUrl2"]
        #Get images:
        img1_resp = requests.get(img1_url)
        img2_resp = requests.get(img2_url)
        img1 = Image.open(BytesIO(img1_resp.content))
        img2 = Image.open(BytesIO(img2_resp.content))

        #Save images:
        img1.save(os.path.join(app.config['UPLOAD_FOLDER'], img1_name))
        img2.save(os.path.join(app.config['UPLOAD_FOLDER'], img2_name))

        # Create file path:
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_name)
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_name)

        prediction = make_prediction(img1_path, img2_path)

        img_prediction = {"img_prediction":prediction}

        # Open a file for writing
        with open(os.path.join(app.config['RESPONSE_FOLDER'], 'img_pred.json'), 'w') as f:
            # Write the dictionary to the file
            json.dump(img_prediction, f)

        return 'prediction saved'

    if request.method == "GET":
        with open(os.path.join(app.config['RESPONSE_FOLDER'], 'img_pred.json'), "r") as f:
            # Load the JSON data from the file
            prediction = json.load(f)
        ### File clean up
        os.remove(os.path.join(app.config['RESPONSE_FOLDER'], 'img_pred.json'))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img1_name))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img2_name))
        return jsonify(prediction)

@app.route("/api/invoke_gpt2", methods = ["GET","POST"])
def chatbot_text_prediction():
    if request.method == "GET":
        # Get uploaded files
        question = request.args.get('msg')
        # #question:
        # global question

        answer = predict(question, 25)
        words=answer[0].split()
        final_ans=[]
        for i in words:
            if (answer[0].count(i)>=1 and (i not in final_ans)):
                final_ans.append(i)

        return ' '.join(final_ans)
        # return answer


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)
