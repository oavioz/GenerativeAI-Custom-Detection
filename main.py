from flask import Flask, jsonify, request
import base64, random, os
import src.AI_detect as AI_detect


app = Flask(__name__)


@app.route("/recognize/images/", methods=['POST'])
def index():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    # save the file
    filename = str(random.randint(0, 1e9)) + file.filename
    filepath = f"./uploads/{filename}"
    file.save(filepath)

    #predict and return the result
    if 'classes' not in request.form: 
        return jsonify(error="Missing classification classes"), 400
    
    res = AI_detect.predict_photo(filepath, request.form.getlist("classes"))
    os.remove(filepath)
    return res 


@app.route("/recognize/videos/", methods=['POST'])
def index(): 
    pass 

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=54362)

