# MSDS 498 Chat Bot Flask App Repo

### To Run:

1. Activate the virtual environment: `source .flaskapp/bin/activate`
2. Install requirements `pip install -r requirements.txt`
2. Start the Flask App: `python3 app.py`
3. Look at App: Go to localhost:5000


### Functionality:

#### SWAM Azure:
- Select `Talk to SWAM Azure` OR...
- Visit: https://msds498swam.azurewebsites.net
- You can ask:
    - Ask a medical question and get a response
    - Submit chest X-Rays for diagnosis
    - Schedule a doctor's appointment
    - Enter `help` to see what other features are available

#### SWAM GPT-2
- Select `Talk to SWAM GPT-2`
- Have a conversation with a Artificial Intelligent Bot trained to answer medical inquiries

#### X-Ray Prediciton
- Select `X-Ray Prediction`
- Provide a frontal and profile chest X-Ray and recieve a cardiopulmonary diagnosis from an AI trained engine.

#### Docker
- Select `Docker`
- You will be directed to DockerHub where you can download an image of our application.

#### Text Echo API
- Select `Text Echo API`
- This will demonstrate the REST API functionality that powers our app by providing an example JSON payload from user generated text. 

#### Text Text REST API
- Select `Test Text REST API`
- This is a Hello World! of REST APIs. A simple JSON payload will be created. 

#### Image Upload Demo
- Select `Image Upload Demo`
- Demonstrate how users can upload images through an HTTP form to the `static/uploads` folder. 