import requests, os, json, sys, shutil 

SERVER_ADDR = f"https://129.159.136.173:54362/"
LOCAL_ADDR = f"https://127.0.0.1:54362/"

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


def send_request_singleTXT(lookdir : str, possible_classes, url):    
    files_mapping = {} 
    found_files = find_files(lookdir)

    for img_path in found_files: 
        files_mapping[img_path] = open(img_path, "rb")

    # Send the image and classes using GET request and print the result. 
    response =  None 
    try: 
        response = requests.post(url, data={
                "classes" : possible_classes, 
                "fast" : "true",
                "filenames" : found_files}, 
                files=files_mapping, verify=False) #Not very good, @TODO: change to keys... 
    
    except Exception as e: 
        print(e)
    
    finally: 
        for f in files_mapping.values(): 
            f.close() 

    if response is None: 
        exit(-1)

    #Parse the results 
    ret = json.loads(response.text)
    ans = [] 
    for key in ret.keys(): 
        ans.append((ret[key], key))
        
    ans.sort()
    ans.reverse() 
    return ans 

            
def escape(path : str): 
    return "_".join(path.split(os.sep))


if __name__ == '__main__':
    if len(sys.argv) != 4: 
        print("Usage: <parent directory>  <string to search>  <dest directory>")
        exit(-1)

    if  not os.path.exists(sys.argv[1]): 
        print("Usage: <parent directory>  <string to search>  <dest directory>")
        exit(-1)

    ans = send_request_singleTXT(sys.argv[1], [sys.argv[2]], SERVER_ADDR + "recognize/images/")

    #Save to the specified folder. 
    if not os.path.exists(sys.argv[3]):  
        os.makedirs(sys.argv[3]) 

    count = 0
    for prob, path in ans: 
        shutil.copyfile(path, os.path.join(sys.argv[3], str(count) + "_" + escape(path[len(sys.argv[1]):])))
        count += 1
