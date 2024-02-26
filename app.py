from datetime import timedelta,datetime
from flask import Flask, render_template, redirect, session, url_for, request, flash, jsonify
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from apscheduler.schedulers.background import BackgroundScheduler
import flask
import time
import cv2
import numpy as np
import tensorflow as tf
import random
import secrets
import pytz
import jwt
import uuid
import os
from flask import Flask
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
socketio = SocketIO(app)
scheduler = BackgroundScheduler()
scheduler.start()

participant_listening_status = {}
participant_sockets = {}
stop_events = {}
participants_data={}

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'alixgizel@gmail.com'
app.config['MAIL_PASSWORD'] = 'bnhw drxr kdsr sjdg'
app.config['MAIL_DEFAULT_SENDER'] = 'alixgizel@gmail.com'
mail = Mail(app)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root2:password@localhost/meeting_app'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    host_email = db.Column(db.String(120), nullable=False)
    code = db.Column(db.String(6), unique=True, nullable=False)

class Participant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    meeting_code = db.Column(db.String(6), nullable=False)
    host_email = db.Column(db.String(120), nullable=False)
    emotions = db.relationship('Emotion', backref='participant', lazy=True)
    join_time= db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    left_time= db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class Emotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(db.Integer, db.ForeignKey('participant.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    emotions_data = db.Column(db.JSON)  


with app.app_context():
    db.create_all()
key_file_path = os.path.join(os.path.dirname(__file__), 'Key 1_12_2024, 1_06_04 AM.pk')  
with open(key_file_path, 'rb') as key_file:
    PRIVATE_KEY = key_file.read()
private_key=serialization.load_pem_private_key(
    PRIVATE_KEY, 
    password=None, 
    backend=default_backend()
)

def generate_jwt():
    now = datetime.utcnow()
    expiration_time = now + timedelta(seconds=7200)  

    payload = {
        "aud": "jitsi",
        "context": {
            "user": {
                "id": str(uuid.uuid4()),
                "name": "",
                "avatar": "",
                "email": "",
                "moderator": "true",
            },
            "features": {
                "livestreaming": "true",
                "recording": "true",
                "transcription": "true",
                "outbound-call": "true",
            },
        },
        "iss": "chat",
        "room": "*",
        "sub": "vpaas-magic-cookie-d8b4d69435b14579ac1d2d3fbd1fa393",
        "nbf": int(now.timestamp()) - 10,
        "exp": int(expiration_time.timestamp()),
    }

    token = jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")
    return token

def generate_code():
    return str(random.randint(000000, 999999))

def send_email(recipient, subject, body):
    message = Message(subject, recipients=[recipient], body=body)
    mail.send(message)

MODEL_PATH = os.path.join('flask', 'model.h5')
CASCADE_PATH = os.path.join('flask', 'haarcascade_frontalface_default.xml')
    
def detect_listening_status_and_emit(name, meeting_code):
    global stop_events
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        cap = cv2.VideoCapture(0)

        last_time = time.time()
        last_emotion = None
        message = None
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error reading frame")
                break
            if stop_events.get(name,False):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            listening = False
            if len(faces) == 0:
                message = "not attending"  # No faces detected
            else:
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]

                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_img = cv2.resize(face_img, (48, 48))
                    face_img = face_img / 255.0
                    face_img = face_img.reshape(1, 48, 48, 1)

                    emotion = model.predict(face_img)[0]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

                    if np.argmax(emotion) < len(categories):
                        emotion_label = categories[np.argmax(emotion)]
                        cv2.putText(frame, emotion_label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        if emotion_label == "Neutral":
                            listening = True
                if time.time() - last_time > 5:
                    message = "listening" if listening else "not listening"
                    if message != last_emotion:
                        last_emotion = message
                        last_time = time.time()
            print(f"Sent listening_status: {message}")
            socketio.emit('listening_status',{
            'status': message,
            'meeting_code': meeting_code,
            'participant_name': name,
            },
            room=meeting_code,
            namespace='/meeting',
            )
            time.sleep(4)
            participant_listening_status.setdefault(name, []).append(message)
    except Exception as e:
        print(f"Error in detect_listening_status_and_emit for participant {name}: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows
        return message
def stop_camera():
    try:
        cv2.destroyAllWindows()  
    except Exception as e:
        print(f"Error in stop_camera: {str(e)}")
def send_participant_message(name, meeting_code, status):
    socketio.emit(
        'participant_status_message',
        {'name': name, 'meeting_code': meeting_code, 'status': status, 'time': datetime.now().strftime("%H:%M:%S")},
        room=meeting_code,
        namespace='/meeting'
    )
    
@application.after_request
def set_response_headers(response):
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    response.headers['Cache-Control'] = 'max-age=31536000, immutable'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/redirect', methods=['POST'])
def redirect_page():
    option = request.form['option']
    if option == 'meeting':
        return redirect(url_for('meeting'))
@app.route('/meeting', methods=['GET','POST'])
def meeting():
    if request.method =='GET':
        return render_template('meeting.html')
    elif request.method =='POST':
        return render_template('meeting.html')

@app.route('/meeting_options', methods=['POST'])
def meeting_options():
    option = request.form['option']
    if option == 'create':
        return render_template('create_meeting.html')
    elif option == 'join':
        return redirect(url_for('join_meeting'))  # Redirect to the join_meeting route

@app.route('/create_meeting', methods=['POST'])
def create_meeting():
    email = request.form['email']
    code = generate_code()
    session['meeting_code'] = code

    # Store the meeting in the database
    meeting = Meeting(host_email=email, code=code)
    db.session.add(meeting)
    db.session.commit()

    # Send email with the code
    send_email(email, 'Meeting Code', f'The meeting code is: {code}')

    flash(f'Code generated and sent to {email}')

    # Render the create_meeting_host.html template with the meeting code
    return render_template('create_meeting_host.html', code=code)

@app.route('/join_meeting_host/<meeting_code>', methods=['GET', 'POST'])
def join_meeting_host(meeting_code):
    if meeting := Meeting.query.filter_by(code=meeting_code).first():
        host_email = meeting.host_email

        # Query the Participant table to get the list of participants for the meeting
        participants = Participant.query.filter_by(meeting_code=meeting_code).all()

        # Prepare a dictionary to store the listening status for each participant
        listening_status_dict = {}

        return render_template('meeting_host.html', meeting_code=meeting_code, host_email=host_email, participants=participants, listening_status_dict=listening_status_dict)
    else:
        return 'Invalid meeting code'
@app.route('/join_meeting', methods=['GET', 'POST'])
def join_meeting():
 try:
    if request.method == 'POST':
        name = request.form['name']
        meeting_code = request.form['meeting_code']

        if meeting := Meeting.query.filter_by(code=meeting_code).first():
            participant = Participant(
                name=name,
                meeting_code=meeting_code,
                host_email=meeting.host_email,
                join_time=datetime.now(pytz.timezone('Asia/Kolkata'))
            )

            db.session.add(participant)
            db.session.commit()

            # Emit a new participant event to update the host's page
            socketio.emit('new_participant', {'name': name, 'index': participant.id}, room=meeting_code, namespace='/meeting')
            socketio.emit('new_participant', {'name': name}, room=meeting_code, namespace='/meeting')
            socketio.start_background_task(detect_listening_status_and_emit, name, meeting_code)

            # Redirect to another route after successful joining
            return redirect(url_for('meeting_joined', meeting_code=meeting_code, host_email=meeting.host_email, name=name, status='joined'))
        else:
            flash('Invalid meeting code')
    return render_template('join_meeting.html')
 except Exception as e:
    print(f"Error in join_meeting: {str(e)}")

@app.route('/meeting_joined/<meeting_code>/<host_email>/<name>/<status>')
def meeting_joined(meeting_code, host_email, name, status):
    participant = Participant.query.filter_by(name=name, meeting_code=meeting_code).first()
    return render_template('meeting_joined.html', meeting_code=meeting_code, host_email=host_email, name=name, status=status)
@app.route('/get_participants/<meeting_code>', methods=['GET'])
def get_participants(meeting_code):
    try:
        # Query the Participant table to get the list of participants for the meeting
        participants = Participant.query.filter_by(meeting_code=meeting_code).all()

        # Prepare a list of dictionaries containing participant information
        participant_list = [{'id': participant.id, 'name': participant.name} for participant in participants]

        # Prepare a dictionary to store listening statuses for each participant
        listening_status_dict = {}
        for key, value in listening_status_dict.items():
            # Check if the value is a Response object
            if isinstance(value, flask.wrappers.Response):
                # Convert Response object to its JSON data
                listening_status_dict[key] = value.json
        # Iterate over participants and update listening status dictionary
        for participant in participants:
            # Call a function to get the listening status for each participant
            listening_status = get_listening_status(participant.name)
            
            # Add the participant's listening status to the dictionary
            listening_status_dict[participant.name] = listening_status
        for key, value in listening_status_dict.items():
            # Check if the value is a Response object
            if isinstance(value, flask.wrappers.Response):
                # Convert Response object to its JSON data
                listening_status_dict[key] = value.json

        # Return the list of participants along with their listening statuses as JSON
        return jsonify({'participants': participant_list, 'listening_status_dict': listening_status_dict})

    except Exception as e:
        print(f"Error in get_participants: {str(e)}")
        # Log the exception, and return an appropriate error response
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_listening_status/<name>')
def get_listening_status(name):
 try:
    if meeting_code := session.get('meeting_code'):
            try:
                # Call the modified function to get the listening status
                message = detect_listening_status_and_emit(name, meeting_code)

                # Emit the listening status to the clients in the meeting room
                socketio.emit(
                    'listening_status',
                    {'status': message, 'meeting_code': meeting_code, 'participant_name': name},
                    room=meeting_code,
                    namespace='/meeting'
                )

                if message:
                    return jsonify({'status': message})
                else:
                    return jsonify({'error':'No listening status found'})
            except Exception as e:
                return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Meeting code not found'})
 except Exception as e:
    return jsonify({'error': f"Error in get_listening_status: {str(e)}"})
    
def calculate_average_listening_status(participant_name):
    statuses = participant_listening_status.get(participant_name, [])
    if not statuses:
        return "No Data"

    total_statuses = len(statuses)
    listening_count = statuses.count("listening")
    not_listening_count = statuses.count("not listening")
    not_attending_count = statuses.count("not attending")

    # Calculate the percentage of each status
    listening_percentage = (listening_count / total_statuses) * 100
    not_listening_percentage = (not_listening_count / total_statuses) * 100
    not_attending_percentage = (not_attending_count / total_statuses) * 100

    # Determine the average listening status
    if listening_percentage >= 50:
        return "listening"
    elif not_listening_percentage >= 50:
        return "not listening"
    elif not_attending_percentage >= 50:
        return "not attending"

# Function to send summary email to the host
def send_summary_email(meeting_code, host_email):
    # Query the Participant table to get the list of participants for the meeting
    participants = Participant.query.filter_by(meeting_code=meeting_code).all()

    listening_status_logs = []
    joining_leaving_logs = []

    for participant in participants:
        # Calculate average listening status for each participant
        participant_status = calculate_average_listening_status(participant.name)
        listening_status_logs.append(f"{participant.name}: {participant_status}")

        # Calculate meeting time
        meeting_time = participant.left_time - participant.join_time

        # Convert meeting time to a human-readable format
        meeting_time_str = str(timedelta(seconds=meeting_time.total_seconds()))

        # Add joining and leaving logs
        joining_leaving_logs.append(f"{participant.name}: meeting_time: {meeting_time_str}")

    subject = 'Meeting Summary'
    body = f'The meeting ({meeting_code}) has ended. Here are the listening status logs:\n\n{" ".join(listening_status_logs)}\n\nJoining and Leaving Logs:\n{" ".join(joining_leaving_logs)}'

    send_email(host_email, subject, body)

@app.route('/end_meeting/<meeting_code>', methods=['POST'])
def end_meeting(meeting_code):
    meeting = Meeting.query.filter_by(code=meeting_code).first()
    if not meeting:
        return 'Invalid meeting code'
    
    host_email = meeting.host_email

    # Call the send_summary_email function to send the summary email
    send_summary_email(meeting_code, host_email)
    socketio.emit('meeting_ended', {'meeting_code': meeting_code}, room=meeting_code, namespace='/meeting')
    return render_template('meeting_ended.html', meeting_code=meeting_code)


@app.route('/leave_meeting/<participant_name>/<meeting_code>', methods=['GET', 'POST'])
def leave_meeting(participant_name, meeting_code):
    if request.method == 'POST':
            participant = Participant.query.filter_by(name=participant_name, meeting_code=meeting_code).first()
            if participant:
                stop_events[participant_name] = True
                participant.left_time = datetime.now(pytz.timezone('Asia/Kolkata')).astimezone(pytz.utc)
                db.session.commit()
                stop_camera()
                return redirect(url_for('leave_meeting', participant_name=participant_name, meeting_code=meeting_code))
            else:
                flash('Participant not found')
    return render_template('leave_meeting.html')

@app.after_request
def set_response_headers(response):
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    response.headers['Cache-Control'] = 'max-age=31536000, immutable'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

if __name__ == '__main__':
    app.secret_key = secrets.token_hex(16)
    socketio.run(app)
