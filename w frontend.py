from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import base64
import logging

app = Flask(__name__)

# Load the trained model and classes
model = tf.keras.models.load_model('final_model.keras')
label_classes = np.load('label_encoder_classes.npy')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def preprocess_keypoints(keypoints, input_shape):
    keypoints = np.array(keypoints).flatten()
    keypoints = keypoints / np.max(keypoints)
    if keypoints.shape[0] != np.prod(input_shape):
        raise ValueError(f"Keypoints shape mismatch: expected {input_shape}, got {keypoints.shape}")
    return keypoints

def predict_label(keypoints):
    input_shape = model.input_shape[1:]
    keypoints = preprocess_keypoints(keypoints, input_shape)
    keypoints = np.expand_dims(keypoints, axis=0)
    prediction = model.predict(keypoints, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    return label_classes[predicted_class[0]], np.max(prediction)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>I-SRAVIA - Sign Language Detection and Speech-to-Text</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
/* Updated with new colors: Dark Purple, Muted Red, Coral Red, Warm Yellow */

body {
    font-family: 'Roboto', sans-serif;
    color: white;
    margin: 0;
    padding: 0;
    height: 100vh;
    position: relative;
    background: linear-gradient(45deg, #57385c, #a75265); /* Dark Purple and Muted Red */
    overflow: hidden;
}

.header {
    text-align: center;
    margin-bottom: 20px;
    background-color: transparent;
    color: white;
    padding: 20px 0;
}

body::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url("https://garden.spoonflower.com/c/8643180/p/f/m/hi4zNy7lP7Xag4inNzLxNBynYi7cN14RCnDB9e5XR1hg0kcE7BMtjgyc/Small_Scale_Tossed_Sign_Language_ASL_Alphabet_Black.jpg");
    background-size: 30%;
    background-position: center;
    background-repeat: repeat;
    filter: opacity(0.05);
    z-index: -1;
}

.container {
    z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 50px;
}

.header h1 {
    margin: 40;
    font-size: 2.5em;
}

.content {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
}

.video-container {
    flex: 1;
    min-width: 300px;
    margin-right: 20px;
    border: 2px solid white;
    padding: 50px;
    border-radius: 10px;
}

#video {
    width: 100%;
    max-width: 640px;
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.controls {
    flex: 1;
    min-width: 300px;
    padding: 50px;
    border: 2px solid white;
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0.05);
}

.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 12px 24px;
    font-size: 16px;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 10px 5px;
    border: 2px solid white;
    color: white;
    background: transparent;
}

.btn-primary {
    background-color: #ec7263; /* Coral Red */
}

.btn-primary:hover {
    background-color: #febe7e; /* Warm Yellow */
}

.btn-warning {
    background-color: #a75265; /* Muted Red */
}

.btn-warning:hover {
    background-color: #57385c; /* Dark Purple */
}

.btn i {
    margin-right: 10px;
}

#result, #speechToTextResult {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
    font-size: 18px;
    border: 2px solid white;
}

.speech-to-text {
    margin-top: 40px;
}

.speech-to-text h2 {
    color: #febe7e; /* Warm Yellow */
}

#micAnimation {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: #ec7263; /* Coral Red */
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 20px auto;
    animation: pulse 1.5s infinite;
    display: none;
}

@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(42, 44, 42, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 10px rgba(236, 114, 99, 0); /* Coral Red */
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(236, 114, 99, 0); /* Coral Red */
    }
}

body {
    animation: bgAnimation 20s linear infinite;
}


    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-sign-language"></i> WELCOME TO I-SRAVIA</h1>
    </div>
    <div class="container">
        <div class="content">
            <div class="video-container">
                <video id="video" autoplay></video>
            </div>
            <div class="controls">
                <button id="startButton" class="btn btn-primary">
                    <i class="fas fa-play"></i> Start Detection
                </button>
                <button id="toggleVoiceBtn" class="btn btn-warning">
                    <i class="fas fa-volume-up"></i> Enable Voice
                </button>
                <div id="result"></div>
                
                <div class="speech-to-text">
                    <h2><i class="fas fa-microphone"></i> Speech-to-Text</h2>
                    <button id="startSpeechToTextButton" class="btn btn-primary">
                        <i class="fas fa-microphone-alt"></i> Start Speaking
                    </button>
                    <div id="micAnimation">
                        <i class="fas fa-microphone" style="color: white; font-size: 24px;"></i>
                    </div>
                    <div id="speechToTextResult"></div>
                </div>
            </div>
        </div>
    </div>
<script>
    let isDetectionActive = false;
    let voiceEnabled = false;
    let detectionInterval;
    let lastPrediction = "";

    document.getElementById('startButton').addEventListener('click', function() {
        if (isDetectionActive) {
            stopDetection();
            this.innerHTML = '<i class="fas fa-play"></i> Start Detection';
            this.classList.remove('btn-danger');
            this.classList.add('btn-primary');
        } else {
            startDetection();
            this.innerHTML = '<i class="fas fa-stop"></i> Stop Detection';
            this.classList.remove('btn-primary');
            this.classList.add('btn-danger');
        }
        isDetectionActive = !isDetectionActive;
    });

    function startDetection() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                document.getElementById('video').srcObject = stream;
                detectionInterval = setInterval(captureFrameAndSend, 1000);
            })
            .catch(error => {
                console.error('Error accessing media devices.', error);
            });
    }

    function stopDetection() {
        clearInterval(detectionInterval);
        const video = document.getElementById('video');
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
    }

    function captureFrameAndSend() {
        const video = document.getElementById('video');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');
        sendFrameToServer(dataURL);
    }

    function sendFrameToServer(imageData) {
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            const resultElement = document.getElementById('result');
            resultElement.textContent = `Prediction: ${data.prediction}`;
            if (voiceEnabled && data.prediction !== lastPrediction) {
                speakPrediction(data.prediction);
                lastPrediction = data.prediction;
            }
        })
        .catch(error => console.error('Error:', error));
    }

    function speakPrediction(text) {
        const speech = new SpeechSynthesisUtterance();
        speech.lang = 'en-US';
        speech.text = text;
        window.speechSynthesis.speak(speech);
    }

    document.getElementById('toggleVoiceBtn').addEventListener('click', function() {
        voiceEnabled = !voiceEnabled;
        this.innerHTML = voiceEnabled ? '<i class="fas fa-volume-off"></i> Disable Voice' : '<i class="fas fa-volume-up"></i> Enable Voice';
    });

    document.getElementById('startSpeechToTextButton').addEventListener('click', startSpeechToText);

    function startSpeechToText() {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
            document.getElementById('micAnimation').style.display = 'block';
        };

        recognition.onend = () => {
            document.getElementById('micAnimation').style.display = 'none';
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            document.getElementById('speechToTextResult').textContent = `You said: ${transcript}`;
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error detected: ', event.error);
        };

        recognition.start();
    }
</script>

</body>
</html>
''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        prediction = "Unknown"
        confidence = 0.0
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                prediction, confidence = predict_label(keypoints)
        
        # Convert confidence to native Python float
        confidence = float(confidence)

        return jsonify({'prediction': prediction, 'confidence': confidence})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'prediction': 'Error', 'confidence': 0.0}), 500

if __name__ == '__main__':
    app.run(debug=True)
