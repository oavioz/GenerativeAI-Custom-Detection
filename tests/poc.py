import requests, os, json


def send_request(lookdir : str, possible_classes):
    # Define the target URL
    url = "http://127.0.0.1:54362/recognize/images/"

    # Send the image and classes using GET request and print the result. 
    response = requests.get(url, params={
            "classes" : possible_classes, 
            "path" : lookdir, 
            "fast" : "True"}) #The default is false, when True is set, frames are proccessed in small jumps
    ret = json.loads(response.text)

    for key in ret.keys(): 
        print(key, ret[key] ,sep=" : ")
            
if __name__ == '__main__':
    send_request(os.path.join("tests", "documents"), ["potato leaf", "A party", "running group", "person with tattoos"])


