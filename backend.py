from flask import Flask, jsonify, request
import random, os, shutil, base64
import src.AI_detect as AI_detect
import src.extract_images as extract_images


app = Flask(__name__)


'''
Receives an image using a post request, runs the model on it, and returns the result. 
It also receives the classes for classification. 
'''
@app.route("/recognize/images/", methods=['POST'])
def recognize_images_post():
    filenames = request.form.getlist("filenames")

    if len(filenames) == 0:
        return jsonify(error="No files found"), 400
    
    if any(fname not in request.files for fname in filenames): 
        return jsonify(error="Files and filenames doesn't match"), 400 
    
    
    if any(request.files[fname] == '' for fname in filenames):
        return jsonify(error="No selected file"), 400
    
    # save the file
    stored = os.path.join(".uploads", str(random.randint(0, 1e9)))
    os.makedirs(stored)
    
    for fname in filenames: 
        filepath = os.path.join(stored, fname)
        if not os.path.exists(os.sep.join(filepath.split(os.sep)[:-1])): 
            os.makedirs(os.sep.join(filepath.split(os.sep)[:-1]))
        request.files[fname].save(filepath)
    
    #Convert to images and find all files 
    basedir = os.path.join(".extracted", str(random.randint(0, 1e9)))
    if not os.path.exists(basedir): 
        os.makedirs(basedir)
    extract_images.main(basedir, stored, (request.form.get("fast").lower() == 'true'))
    image_paths = extract_images.find_files(basedir)

    #Run the model and returns the results. 
    ret = {} 
    if 'classes' not in request.form:
        shutil.rmtree(stored)
        shutil.rmtree(basedir) 
        return jsonify(error="Missing classification classes"), 400
    
    if len(request.form.getlist("classes")) == 1: 
        ret = AI_detect.predict_text(image_paths, request.form.getlist("classes"), len(basedir) + len(stored) + 2)
        shutil.rmtree(stored)
        shutil.rmtree(basedir)
        return ret 

    for img in image_paths: 
        ret[img[len(basedir) + len(stored) + 2:]] = AI_detect.predict_photo(img, request.form.getlist("classes"))

    shutil.rmtree(stored)
    shutil.rmtree(basedir)
    return ret 


'''
Instead of sending the file, sends a path to a directory 
@TODO: support url instead of path, or add a module that downloads the directory... 
When only one class is received, the code sends a probability distribution on the images instead. 
'''
@app.route("/recognize/images/", methods=['GET'])
def recognize_images_get():
    path = request.args.get("path")
    if path is None: 
        return jsonify("Path wasn't received"), 400 
    
    fast = (request.args.get("fast").lower() == 'true') 
    possible_classes = request.args.getlist("classes")

    #Converts everything to images and saves it inside basedir. 
    basedir = os.path.join(".extracted", str(random.randint(0, 1e9)))
    os.makedirs(basedir)
    extract_images.main(basedir, path, fast)

    #Runs the model on every image, and returns a json
    #The json is of the form {image_name : prediction}
    image_paths = extract_images.find_files(basedir)
    ret = {} 
    
    if len(possible_classes) == 1: 
        ret = AI_detect.predict_text(image_paths, possible_classes[0], len(basedir) + 1)
        shutil.rmtree(basedir)
        return ret 

    for img in image_paths: 
        ret[img[len(basedir) + 1:]] = AI_detect.predict_photo(img, possible_classes)

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

    if len(que) > 400: 
        return jsonify(error="Query exceeds maximal length (400)"), 400    
    
    ans = AI_detect.query_text(que)
    return ans


if __name__ == "__main__":
    app.run(ssl_context=("keys/public.crt", "keys/private.key"), host="0.0.0.0", port=54362)

