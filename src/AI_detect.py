import os, shutil, time
import src.extract_images as extract_images
import torch, clip
from PIL import Image
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


'''Uses openAI's CLIP to predict the labels'''
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
