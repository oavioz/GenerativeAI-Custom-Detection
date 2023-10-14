from flask import Flask, jsonify, request
import random, os, shutil
import src.AI_detect as AI_detect
import src.extract_images as extract_images

app = Flask(__name__)

'''
Receives an image using a post request, runs the model on it, and returns the result. 
It also receives the classes for classification. 
'''
@app.route("/recognize/images/", methods=['POST'])
def recognize_images_post():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    #Otherwise, the server collapses when uploads doesn't exists 
    if not os.path.exists("uploads"): 
        os.makedirs("uploads")
    
    # save the file
    filename = str(random.randint(0, 1e9)) + file.filename
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    #predict and return the result
    if 'classes' not in request.form: 
        return jsonify(error="Missing classification classes"), 400
    
    res = AI_detect.predict_photo(filepath, request.form.getlist("classes"))
    os.remove(filepath)
    return res 


'''
Instead of sending the file, sends a path to a directory 
@TODO: support url instead of path, or add a module that downloads the directory... 
'''
@app.route("/recognize/images/", methods=['GET'])
def recognize_images_get():
    path = request.args.get("path")
    if path is None: 
        return jsonify("Path wasn't received"), 400 
    
    fast = (request.args.get("fast").lower() == 'true') 
    possible_classes = request.args.getlist("classes")

    #Converts everything to images and saves it inside basedir. 
    basedir = os.path.join("extracted", str(random.randint(0, 1e9)))
    os.makedirs(basedir)
    extract_images.main(basedir, path, fast)

    #Runs the model on every image, and returns a json
    #The json is of the form {image_name : prediction}
    images_path = extract_images.find_files(basedir)
    ret = {} 

    for img in images_path: 
        ret[img[len(basedir):]] = AI_detect.predict_photo(img, possible_classes)

    shutil.rmtree(basedir)
    return ret 


'''
Sends text query to chatgpt 3.5, and returns the result. 
'''
@app.route("/query/chatgpt/", methods=['GET'])
def query_chatgpt(): 
    que = request.args.get("query") 
    if que is None: 
        return jsonify(error="Missing Query"), 400 

    if len(que) > 100: 
        return jsonify(error="Query exceeds maximal length (100)"), 400    
    
    ans = AI_detect.query_text(que)
    return ans


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=54362)

