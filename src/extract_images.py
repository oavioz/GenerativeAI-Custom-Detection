import cv2, os, shutil  


#When fast=True, only  1/FAST_PAR frames are proccessed. 
FAST_PAR = 15 

'''
split the video to frames, and saves all of them as jpg. 
src: https://note.nkmk.me/en/python-opencv-video-to-still-image/'''
def save_all_frames(video_path, dir_path, basename, ext='jpg', fast = False):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            n += 1
            if fast and n % FAST_PAR != 0: 
                continue

            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
        else:
            return


'''recursive look for videos from a folder
returns a list of all the files (path, string)'''
def find_files(src_folder : str) -> list: 
    if os.path.isfile(src_folder): 
        return [src_folder]
    
    directories = [src_folder]
    found = []
    while len(directories) > 0: 
        now = directories[-1]
        directories.pop()

        for filename in os.listdir(now): 
            f = os.path.join(now, filename) 
            if os.path.isfile(f):
                found.append(f)
            else: 
                directories.append(f)
    return found


'''helper function that creates the image path'''
def find_path(vid : str): 
    dir = vid.split(os.sep)[:-1]
    dir = os.sep.join(dir)
    filename = vid.split(os.sep)[-1]
    return dir, filename


'''extract videos to frames, doesn't modify pictures'''
def main(basedir : str, vid_dir : str, fast=False) -> None: 
    found_vids = find_files(vid_dir)
    for vid in found_vids: 
        dir, filename = find_path(vid)

        if vid.split(".")[-1] in ["mp4", "mov", "wmv", "avi"]: #Common movies formats
            save_all_frames(vid, os.path.join(basedir, dir), filename, 'jpg', fast) 
        
        elif vid.split(".")[-1] in ["jpeg", "png", "jpg"]: 
            if not os.path.exists(os.path.join(basedir, dir)): 
                os.makedirs(os.path.join(basedir, dir)) 
            shutil.copyfile(vid, os.path.join(basedir, dir, filename))


'''will convert all the videos to images'''
if __name__ == '__main__': 
    main() 
