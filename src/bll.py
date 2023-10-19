from dal import *
import urllib.request
from s3_download import *
from mocap_if import *

import random

FOLDER_PATH = "/data/imgs"


async def indexFiles(data):
    results = []
    for file in data:
       results.append(await indexFile(file))
    return results


async def indexFile(data):
    return await indexFileLogic(data.ImgURL)

async def indexFileLogic(imgUrl):

    #check if file exists in db
    file_rec = await getFileByURL(imgUrl)

    if file_rec is not None:
        print("File already exists in db",file_rec)
        return file_rec
   
    #if not exists - download file

    if imgUrl.startswith("https"):
        print("Downloading from URL https")
        # Extract the file name from the URL
        file_name = os.path.basename(file_url)
        # Generate a random number and append it to the file name
        random_number = random.randint(1, 100000)  # Adjust the range as needed
        file_name_with_random = f"{os.path.splitext(file_name)[0]}_{random_number}{os.path.splitext(file_name)[1]}"
        file_path = os.path.join(folder_path, file_name_with_random)
        # Download the file and save it to the specified path
        urllib.request.urlretrieve(data.ImgURL, file_path)
        
    elif imgUrl.startswith("S3") or imgUrl.startswith("s3"):
        print("Downloading from s3")
        file_path = download_file_from_s3(imgUrl,"/data/imgs")
        print("file_path",file_path)


    if file_path is None:
        return {"error": "file not found"}

    
    add_records([{"img_url":file_path}])

    await insertRecord({"img_url":imgUrl,"img_path":file_path})

    return {"success": "file indexed"}




async def search_with_text(text):
    print("searching text",text)
    final_result = []
    results = search_text(text)
    for i in range(1,100):
    # for result in results:
        result = results[i]
        print("result",result)
        # img_url = 
        # if 

        file_rec = await getFileByPath("/"+result["img_url"]);
        print("s3_url",file_rec)
        if file_rec:
            result["public_img_url"] = await getPresingedURL(file_rec["img_url"])
            final_result.append(result)
    print("final_result",final_result)
    return final_result