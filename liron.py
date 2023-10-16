from flask import Flask, jsonify, request
import random, os, shutil, base64, json 
import src.AI_detect as AI_detect
import src.extract_images as extract_images


def recognize_images_post(basedir, classes):
    
    image_paths = extract_images.find_files(basedir)

    if len(classes) == 1: 
        ret = AI_detect.predict_text(image_paths, classes, 0)
        return ret 

    for img in image_paths: 
        ret[img] = AI_detect.predict_photo(img, classes)

    return ret 


if __name__ == '__main__': 
    ret = recognize_images_post("red", ["A woman with a cast on her leg"])
    
    ans = [] 
    for key in ret.keys(): 
        ans.append((ret[key], key))
        
    ans.sort()
    ans.reverse() 

    for i in range(min(20, len(ans))): 
        print(ans[i][1])
