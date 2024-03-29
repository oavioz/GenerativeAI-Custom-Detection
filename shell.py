if __name__ == "__main__": print("Wait while the model is loading...")
import requests
import os, json, shutil
import src.extract_images as extract_images
import src.AI_detect as AI_detect

def match_and_parse(filenames, pm): 
    res = AI_detect.predict_text(filenames, pm) 
    return [res[filenames[i]] for i in range(len(filenames))]


def proccess_request(basedir : str, prompts : list): 
    filenames = extract_images.find_files(basedir) 

    count = [0 for i in range(len(filenames))]
    for pm in prompts: 
        now = match_and_parse(filenames, pm)
        for i in range(len(count)): 
            count[i] += now[i] 

    for i in range(len(count)): 
        count[i] /= len(prompts)
        count[i] *= 100 

    ans = list(zip(count, filenames))
    ans.sort()
    ans.reverse()
    return ans 


def main(): 
    basedir = input("Enter base directory: ")
    if not os.path.exists(basedir): 
        print("Path not found")
        exit(-1)

    count = 0
    try: 
        count = int(input("Enter amount of prompts: "))
        if count <= 0: 
            raise Exception
    except: 
        print("Must be a positive integer")
        exit(-1)
    
    prompts = []
    print("Enter prompts:")
    for _ in range(count): 
        prompts.append(input("-> "))

    #process the request, run the ai model and return a parsed result.
    ans = proccess_request(basedir, prompts)    
    

    print("Finished processing, move results to a new directory?")
    move = input("yes/no ")
    if move.lower().find("yes") != -1: 
        move = input("dir: ") 
    else: 
        move = None 
    if move is not None and not os.path.exists(move):     
        os.makedirs(move)

    #While user requests, print a new detection. 
    count = 0
    for confid, path in ans: 
        print(round(confid / ans[0][0], 8), " : ", path, end=" ")

        #to make easier to scp to local
        if move is not None:
            shutil.copyfile(path, os.path.join(move, str(count)) + "." + path.split(".")[-1]) 
            count += 1
        
        input() 

            

if __name__ == '__main__': 
    main() 