import torch
import numpy as np
import cv2
from flask import Flask, render_template, Response, request

app = Flask(__name__)

default_video_path = 'shooting1.mp4'
uploaded_video_path = None

# Add your YOLOv5 model loading here
model = torch.hub.load("yolov5", "custom", path="yolov5/runs/train/exp/weights/last.pt", source="local")

def perform_object_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()

        results = model(frame)

        # Convert the results to an image that can be displayed by OpenCV
        rendered_frame = np.squeeze(results.render())

        # Encode the frame as JPEG before streaming it to the browser
        _, jpeg_frame = cv2.imencode('.jpg', rendered_frame)

        # Yield the frame in bytes format to the Flask web server
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')

        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if uploaded_video_path:
        return Response(perform_object_detection(uploaded_video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(perform_object_detection(default_video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['POST'])
def upload_video():
    global uploaded_video_path
    uploaded_file = request.files['video_file']
    if uploaded_file:
        # Save the uploaded video to a temporary file
        uploaded_video_path = 'uploaded_video.mp4'
        uploaded_file.save(uploaded_video_path)
    return render_template('index.html')  # Redirect back to the index page

if __name__ == '__main__':
    app.run(debug=True)
