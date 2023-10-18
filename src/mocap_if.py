from typing import List
import extract_images
import json, os 
import open_vocabulary_image_classification as ovic
import extract_images

JSON_PATH = 'server_data/info.json'
PRINT = False

def add_records(rec_list : List[dict]) -> None:
    if PRINT: print("Building new files...")
    ovic.handle_request("build", files=[d['img_url'] for d in rec_list])

def search_text(text: str) -> List[dict]:
    if PRINT: print("Searching the text: \"{}\"".format(text)) 
 
    ans = []
    ovic.handle_request("search", txt=text)
    with open("results.txt", "r") as f: 
        while True: 
            data = f.readline().split(";")
            if len(data) == 1: 
                break
            ans.append({"score": data[0], "img_url" : data[1].strip()})
    return ans 


def query_existence(rec : dict):
    return False 


def main(): 
    act = input("action ")
    if act[0] == "b": 
        rec_list = [] 
        count = int(input("Enter amount of files: ")) 
        if count == 0: 
            rec_list = extract_images.find_files("/home/ubuntu/red/imgs")
            rec_list = [{"img_url" : l} for l in rec_list]
        for _ in range(count): 
            rec_list.append({"img_url" : input("->")})
        add_records(rec_list)
        print("Done build")

    if act[0] == "s": 
        print(str(search_text(input("txt: "))[:10]))
        print("Done search!") 

if __name__ == "__main__": 
    main() 
