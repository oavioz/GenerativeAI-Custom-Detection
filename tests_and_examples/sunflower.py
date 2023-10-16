import requests
import os, json  
'''
Note: this file assumes that documents/sunflower exists. 
Since the file is large, we don't store it on github. 
'''

'''
Same as src.extract_images.find_files(...)
Another copy since the client doesn't need to save the whole project. 
recursive look for videos from a folder
returns a list of all the files (path, string)'''
def find_files(src_folder : str) -> list: 
    if os.path.isfile(src_folder): 
        return [src_folder]
    
    directories = [src_folder]
    found = []
    while len(directories) > 0: 
        now = directories[-1]
        directories.pop()

        for filename in os.listdir(now): 
            f = os.path.join(now, filename) 
            if os.path.isfile(f):
                found.append(f)
            else: 
                directories.append(f)
    return found


def send_request(lookdir : str, possible_classes, url = "https://127.0.0.1:54362/recognize/images/"):
    files_mapping = {} 
    found_files = find_files(lookdir)[::20]

    for img_path in found_files: 
        files_mapping[img_path] = open(img_path, "rb")

    # Send the image and classes using POST request and print the result. 
    response = requests.post(url, data={
            "classes" : possible_classes,
            "fast" : "false",
            'filenames' : found_files}, 
            files=files_mapping, verify=False)
    
    return response.text

            
if __name__ == '__main__':
    possible_classes = ["Sunflower with light green to yellow angular spots on the upper surfaces of leaves", 
                        "Sunflower with lighting of leaves, petioles, blossoms and stems", 
                        "Sunflower without any disease", 
                        "Sunflower with scarring signs"]
    
    #Receives confidence distribution for every image 
    jsn = send_request(os.path.join("documents", "Sunflower"), possible_classes)
    jsn = json.loads(jsn)
    
    #Find the disease
    substr = ["downy", "gray", "fresh", "scar"]
    count = [0 for _ in range(len(possible_classes))]
    for key in jsn.keys(): 
        best = 0 
        for i in range(len(possible_classes)): 
            if jsn[key][possible_classes[i]] > jsn[key][possible_classes[best]]: 
                best = i 

        print(substr[best], ":", jsn[key][possible_classes[best]], ":", key)
        
                


    
    #ask ChatGPT for explanation. 
    #response = requests.get(url="http://127.0.0.1:54362/query/chatgpt/", params={"query" : possible_classes[best] + ". please list the treatment proccess, be specified on pesticise, fertilize, irigation and weather."})
    #response = json.dumps(response.text)
    #print(response)
        




    
    

