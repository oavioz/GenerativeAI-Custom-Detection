
## Running:

To run the backend, run: 
`python3 main.py`
When the server is up, enter the **tests_and_examples**  folder to run examples... 

## The project structure: 
`main.py` : The backend server 
`src/AI_detect.py`:  library that analyze text/pictures/videos using AI
`src/extract_images.py` library for handling files and directories. 

The `main.py` will create the folders "extracted" and "uploads", 
Will store images (possibly after conversion or frames), call functions from `src/AI_detect.py` and returns the result, after parsing. 