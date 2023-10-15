import requests, os, json


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


def send_request_singleTXT(lookdir : str, text_search : str, url = f"http://127.0.0.1:54362/recognize/images/"):    
    files_mapping = {} 
    found_files = find_files(lookdir)
    for img_path in found_files: 
        files_mapping[img_path] = open(img_path, "rb")

    # Send the image and classes using GET request and print the result. 
    try: 
        response = requests.post(url, data={
                "classes" : [text_search], 
                "fast" : "true", #The default is false, when True is set, frames are proccessed in small jumps
                "filenames" : found_files}, 
                files=files_mapping)
    
    except: 
        print("UNKNOWN FAILURE")
    
    finally: 
        for f in files_mapping.values(): 
            f.close() 

    ret = json.loads(response.text)
    ans = [] 
    for key in ret.keys(): 
        ans.append((ret[key], key))
        
    ans.sort()
    ans.reverse() 
    return ans #returns the file sorted by probability check. 
        

def photo_analyzer(filename, text_search, url = f"http://127.0.0.1:54362/recognize/images/"): 
    ans = send_request_singleTXT(filename, text_search, url)
    #write log using chatgpt
    return ans


if __name__ == '__main__':
    print(photo_analyzer(os.path.join("documents", "potatoes"), "a potato leaf with early blight"))  

