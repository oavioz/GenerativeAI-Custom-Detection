from typing import List
import json
import open_vocabulary_image_classification as ovic
import extract_images

IMAGE_PATH = '/home/ubuntu/red/imgs/'
RED_ENC_PATH = '/data/encs/'

def add_records(rec_list : List[dict]):
    print("Building new files...")
    ovic.handle_request("build", files=[d['img_url'] for d in rec_list])

def search_text(text: str):
    print("Searching the text: \"{}\"".format(text)) 
    return ovic.handle_request("search", txt=text)


def query_existence(rec : dict):
    print("found")
    return True


def main(): 
    act = input("action")
    if act == "bulid": 
        filenames = extract_images.find_files(IMAGE_PATH)
        add_records({"img_url" : f} for f in filenames) 
    
    if act == "search": 
        print(str(search_text(input("txt: ")))[:100])

    print("Done!") 

if __name__ == "__main__": 
    main() 