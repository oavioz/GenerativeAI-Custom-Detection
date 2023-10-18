from typing import List
import json
import open_vocabulary_image_classification as ovic
import extract_images

IMAGE_PATH = '/home/ubuntu/red/imgs/'
RED_ENC_PATH = '/data/encs/'
PRINT = False

def add_records(rec_list : List[dict]):
    if PRINT: print("Building new files...")
    ovic.handle_request("build", files=[d['img_url'] for d in rec_list])

def search_text(text: str):
    if PRINT: print("Searching the text: \"{}\"".format(text)) 
    return ovic.handle_request("search", txt=text)


def query_existence(rec : dict):
    if PRINT: print("found")
    return True


def main(): 
    act = input("action")
    if act == "bulid": 
        files = [] 
        count = int(input("Enter amount of files: ")) 
        for _ in range(count): 
            files.append({"img_url" : input("->")})
    add_records(files)
    
    if act == "search": 
        print(str(search_text(input("txt: ")))[:100])

    print("Done!") 

if __name__ == "__main__": 
    main() 