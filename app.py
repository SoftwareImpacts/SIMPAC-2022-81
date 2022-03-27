from flask import Flask,render_template,request,jsonify
import os
import json
import cv2
from werkzeug.utils import secure_filename
from backend.mri.mri_model import *
from backend.wsi.wsi_model import *


app = Flask(__name__)

@app.route("/")
@app.route("/home")
def hello_world():
    return render_template('home.html')


@app.route("/signin")
def signin():
    return render_template('signin.html')

@app.route("/signup")
def signup():
    return render_template('signup.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/gradCAM")
def gradCAM():
    return render_template('gradCAM.html')

@app.route("/mri")
def mri():
    return render_template('mri.html')

@app.route("/wsi")
def wsi():
    return render_template('wsi.html')

def clear_directory(path):
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    print(" Directory Cleared")


@app.route('/gradcam', methods=['POST','GET'])
def get_post_javascript_data():
    clear_directory("static/upload/mri")

    UPLOAD_FOLDER = 'static/upload'  
    app.config['UPLOAD_PATH'] = UPLOAD_FOLDER + "/mri" 
    formData = request.files.getlist("files[]")
    for f in formData:
        f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
    
    classification_class, probabaility, new_file_name = main()
    
    class_name=""

    if classification_class==0:
        class_name = "Glioblastoma"
    elif classification_class==1:
        class_name = "Astrocytoma"
    else:
        class_name = "Oligodendroglioma"

    img = cv2.imread( "static/upload/result/"+new_file_name)
    cropped_image = img[300:2700, 0:600]
    cv2.imwrite("static/upload/result/"+new_file_name, cropped_image)

    
    return render_template("gradCAM.html",image = "upload/result/"+new_file_name, classname = class_name, probabaility = probabaility)
    # return jsonify({"result":"upload/result/gradCAM.png"})
    # return json.loads(jsdata)[0] 

@app.route('/wsi_result', methods=['POST','GET'])
def get_wsi_results():
    # clear_directory("static/upload/wsi/Data_Directory")
    # UPLOAD_FOLDER = 'static/upload'  
    # app.config['UPLOAD_PATH'] = UPLOAD_FOLDER + "/wsi" 
    # formData = request.files.getlist("files[]")
    # for f in formData:
    #     f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
    
    # y = wsi_main()
    y = [0.9929837, 0.00489598, 0.00212026]
    # y = [round(x,2) for x in y]
    probability = 0
    tumor = ""
    if y[0] > y[1]:
        if y[0] > y[2]:
            tumor = "Glioblastoma"
            probability = y[0] * 100
        else:
            tumor = "Oligodendroglioma"
            probability = y[2] * 100
    else:
        if y[1] > y[2]:
            tumor = "Astrocytoma"
            probability = y[1] * 100
        else:
            tumor = "Oligodendroglioma"
            probability = y[2] * 100
    probability = math.trunc(probability)
    orginal_image = "upload/result/wsi/HEATMAP_OUTPUT/Unspecified/CPM19_TCIA02_179_1_orig_0.jpg"
    heatmap_image = "upload/result/wsi/HEATMAP_OUTPUT/Unspecified/CPM19_TCIA02_179_1_0.5_roi_0_blur_0_rs_1_bc_0_a_0.4_l_-1_bi_0_-1.0.jpg"
    return render_template("wsi_results.html", orginal_image=orginal_image, heatmap_image=heatmap_image, classname = tumor, probability=probability)

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0',debug=True)