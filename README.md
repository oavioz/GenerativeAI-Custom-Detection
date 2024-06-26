
## Running:

To run the backend, run: 
`python3 backend.py`
When the server is up, enter the **tests_and_examples**  folder to run examples... 

## The project structure: 
`backend.py` : The backend server 
`src/AI_detect.py`:  library that analyze text/pictures/videos using AI
`src/extract_images.py` library for handling files and directories. 

The `backend.py` will create the folders "extracted" and "uploads", 
Will store images (possibly after conversion or frames), call functions from `src/AI_detect.py` and returns the result, after parsing. 

## Shell tool 
shell_tool.py, allows sorting images according to prompts. 
Before running, it's recomended to add the following alias (sorry windows): 
`alias aitool="python3 shell_tool.py`
Run: 
`aitool <parent directory>  <string to search>  <dest directory>`
It is recommended to use "sorted_files" as a prefix for dest-directory
At the moment, videos are transformed to frames. 

## HTTP request: 
using python requests: 

response = requests.post(url, data={
        "classes" : possible_classes, 
        "fast" : "true",
        "filenames" : found_files}, 
        files=files_mapping, verify=False) 

"classes":  a list of the classes you want CLIP to classify
"fast": by default false, if set to true, the analysis on videos is done every 15 frames, 
which is faster but may lead to data loss. 

"filanames" : a list of all the filenames
"files" all the files you want to transfer, a dictionary in the following format: 
{"filename" : file}

**example**: 
<pre>
1. def send_request(filename : str, possible_classes, url : str):\
2.    # Define the target URL\
3.    # Send the image and classes using POST request and print the result. \
4.    with open("mold.png", "rb") as pic:  \
5.        response = requests.post(url, data={\
6.                "classes" : ["A sunflower leaf with gray mold", "A healthy sunflower or sunflower leaf"],\
7.                "fast" : "false",\
8.                'filenames' : ["mold.png"]}, \
9.                files={"mold.png" : pic})\
10.       print(response.text)
</pre>