from typing import List
import json


IMAGE_PATH = '/home/ubuntu/red/imgs/'
RED_ENC_PATH = '/data/encs/'

def add_records(rec_list : List[dict]):
    print(rec_list)

def search_text(text: str):
    print("Searching the text: \"{}\"".format(text)) 
    return [{'img_url': 'blabla', 'score': 1}, {}, {}]


def query_existence(rec : dict):
    print("found")
    return True
