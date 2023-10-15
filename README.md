
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
Before running, add the following alias: 
`alias aitool="python3 shell_tool.py`
Run: 
`aitool <parent directory>  <string to search>  <dest directory>`
At the moment, videos are transformed to frames. 
