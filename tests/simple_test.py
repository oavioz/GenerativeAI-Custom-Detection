import requests

tattoo_or_running = ["A person with a black tattoo on it's hand", "A running group"]

def send_request(filename : str, possible_classes = tattoo_or_running):
    # Define the target URL
    url = "http://127.0.0.1:54362/recognize/images/"

    # Send the image and classes using POST request and print the result. 
    with open(filename, "rb") as pic:  
        response = requests.post(url, data={
                "classes" : possible_classes}, 
                files={"file" : pic})
        print(response.text)
            
if __name__ == '__main__':
    send_request("documents/tattoo.jpeg") #A bald person with many tattoos 
    send_request("documents/running.jpg") #running group
    send_request("documents/running2.jpg") #Here, one of the runners has a tattoo


