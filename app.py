from ssl import AlertDescription
from pip import main
from flask import Flask, flash, request, redirect, url_for, render_template
import os
from numpy import roots
from werkzeug.utils import secure_filename
import face_recognition as face
import numpy as np
import cv2

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html',main_pic = 'static/uploads/menu_pic.png')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        global G_name 
        G_name = filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename =filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)





@app.route('/process',methods = ['POST'] )
def process():
    person_image = face.load_image_file('static/uploads/'+G_name)
    person_encode = face.face_encodings(person_image)[0]

    face_location = []
    face_encoding = []
    face_name = []
    face_percent = []
    keepframe = True
    known_face_encoding = [person_encode]
    known_face_name = ["FOUND PERSON"]
    cam_num = int(request.form["number"])

    #cam_new_num = cam_new_num
    i = 0
    while i != 1:
        video_cap = cv2.VideoCapture(cam_num)
        ret, frame = video_cap.read()
        face_name = []
        face_percent = []
        if keepframe:
            face_location = face.face_locations(frame)
            face_encoding = face.face_encodings(frame,face_location, model="cnn")

            for face_encoding in face_encoding:
                face_distances = face.face_distance(known_face_encoding,face_encoding)
                best = np.argmin(face_distances)
                face_percent_value = 1 - face_distances[best]

                if face_percent_value >= 0.6:
                    name = known_face_name[best]
                    percent = round(face_percent_value*100,2)
                    face_percent.append(percent)
                else:
                    name = "NOT FOUND"
                    face_percent.append(0)
                face_name.append(name)  

                if name == "FOUND PERSON":
                    cv2.imwrite("static/uploads/show.jpg",frame)
                    print(name + str(percent))
                else:
                    print("a")
                   # cv2.imwrite("static/uploads/show.jpg",frame)########
        keepframe = not keepframe
        i = i + 1
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], G_name))
    #return name + str(percent)
    if name == "FOUND PERSON":
        return render_template('index.html',name = 'show.jpg')
    else:
        return render_template('index.html',main_pic = 'static/uploads/PERSON NOT FOUND.jpg')
    



# @app.route('/test',methods=['POST'])
# def method_test():
#     print(G_name)
#     os.remove(os.path.join(app.config['UPLOAD_FOLDER'], G_name))
#     render_template('index.html')
#     return render_template('index.html')

if __name__ == "__main__":
    app.run()


#py -3 -m venv .venv