import os, shutil, random, torch, clip, openai
import src.extract_images as extract_images
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) #@TODO compare models

#This key has a limit of 10$, may not be enough for production. 
openai.api_key = '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' 


'''Uses openAI's CLIP to predict the labels
Classifier'''
def predict_photo(path : str, possible_classes : list) -> dict: 
    assert os.path.isfile(path)
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    text = clip.tokenize(possible_classes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    res = {} 
    for i in range(len(possible_classes)): 
        res[possible_classes[i]] = float(probs[0][i])
    return res 


'''
Splits the video to frames, runs "predict_photo" and avarage the results.
@TODO: maybe, average isn't the best option, since information that appears in a single frame 
won't affect as much. *maybe l_2 norm is better 
'''
#At the moment, this function has no use. 
def predict_video(path, possible_classes : list) -> dict:
    assert os.path.isfile(path)
    mid = "__mid_images" + str(random.randint(0, 1e9)) 
    extract_images.main(mid, path, True)
    img_analysis = extract_images.find_files(mid) 

    detected = {} 
    counter = 0
    for img_path in img_analysis: 
        counter += 1
        now = predict_photo(img_path, possible_classes)
        for key in now.keys(): 
            detected[key] = detected.get(key, 0) + now[key] 
    shutil.rmtree(mid)

    if counter == 0: 
        return {}     
    for key in detected.keys(): 
        detected[key] /= counter

    return detected


'''
Receives a batch of images and a text query, and returns for every picture the prob that it corresponds to the query. 
'''
def predict_text(image_paths : list, text : str) -> dict: 
    assert all(os.path.isfile(path) for path in image_paths) 
    
    images = torch.stack([preprocess(Image.open(path)).to(device) for path in image_paths])
    text = clip.tokenize(text).to(device)
    

    with torch.no_grad():
        image_features = model.encode_image(images) 
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()

    ret = {} 
    for i in range(len(images)): 
        ret[image_paths[i]] = float(probs[0][i])
    return ret  


'''
Sends a question to chatgpt and returns the response. 
'''
def query_text(que : str) -> str: 
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": que}],
            temperature=0.8)

    return response