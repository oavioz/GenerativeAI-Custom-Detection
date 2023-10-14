import requests
import os 

possible_classes = []


def send_request(que : str) -> None:
    # Define the target URL
    url = "http://127.0.0.1:54362/query/chatgpt/"

    response = requests.get(url=url, params={"query" : que})
    print(response)
            
if __name__ == '__main__':
    send_request("How old is Donald Trump?")

