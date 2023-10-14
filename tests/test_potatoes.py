import requests
import os 

possible_classes = []


def send_request(filename : str, possible_classes):
    # Define the target URL
    url = "http://127.0.0.1:54362/recognize/images/"

    # Send the image and classes using POST request and print the result. 
    with open(filename, "rb") as pic:  
        response = requests.post(url, data={
                "classes" : possible_classes}, 
                files={"file" : pic})
        print(response.text)
            
if __name__ == '__main__':
    send_request(os.path.join("documents", "potatoes", "early_blight.jpg"), possible_classes)
    send_request(os.path.join("documents", "potatoes", "healthy.jpeg"), possible_classes)
    send_request(os.path.join("documents", "potatoes", "late_blight.jpg"), possible_classes)
    send_request(os.path.join("documents", "potatoes", "mosaic.jpg"), possible_classes)
    send_request(os.path.join("documents", "potatoes", "septoria_leaf.jpg"), possible_classes)

