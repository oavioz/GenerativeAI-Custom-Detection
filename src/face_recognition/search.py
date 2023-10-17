from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from face_detect import FaceDetector
from tqdm import tqdm

def l1(x, y):
    return (x-y).abs().sum(-1)

def l2(x, y):
    return ((x-y)**2).sum(-1)

def cosine(x, y):
    return -nn.CosineSimilarity(dim=-1)(x, y)

def entropy_func(x):
    return -torch.sum(x * torch.log(x + 1e-10))

def search_image_in_folder(images_folder = "/home/ubuntu/Nadav/data/blue", 
                           search_path = "/home/ubuntu/red/imgs/noa_3.jpg", 
                           topk = 5, 
                           detect_red = True, 
                           model_name = "casia-webface", 
                           metric = "l2", 
                           agg = "mean", 
                           face_detector_path = "saved_models/yolov8n-face.onnx", 
                           device = torch.device("cpu"), 
                           verbose = False):
    """
    Returns topk matches of search_path in images_folder, and their scores.

    Parameters:
        images_folder (str): The folder path of images (blue or red).
        search_path (str): The image/directory path to search (blue or red), or list of such for multiple persons.
        topk (int): How many to return.
        detect_red (bool): Whereas to face detect red images.
        model_name (str): Face recognition model: "casia-webface", "vggface2".
        metric (str): Correlation metric: "l1", "l2", "cosine".
        agg (str): Aggregation function: "min", "mean".
        face_detector_path (str): Face detection model: "saved_models/yolov8n-face.onnx".
        device (torch.device): Device, what can I say?
        verbose (bool): Print or not.

    Returns:
        float: Matched folder/images names, and their correlation scores.
    """

    face_detector = FaceDetector(face_detector_path)

    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),  # Convert PIL Image to a Torch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to -1 ~ 1
    ])

        # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained=model_name).eval().to(device)

    #print(f"model_name: {model_name} | metric: {metric} | agg: {agg}\n")

    if metric == "l1":
        metric_func = l1
    elif metric == "l2":
        metric_func = l2
    elif metric == "cosine":
        metric_func = cosine

    if agg == "min":
        agg_func = lambda t:t.min()
    elif agg == "mean":
        agg_func = lambda t:t.mean()

    persons = []
    all_scores = []

    with torch.no_grad():

        if not isinstance(search_path, list):
            search_path = [search_path]
        
        all_searches = []
        for search_path_ in search_path:
            if os.path.isfile(search_path_):
                search_images = [search_path_]
            else:
                search_images = [f"{search_path_}/{image_name}" for image_name in os.listdir(search_path_)]
            all_searches.append(search_images)
        
        all_searches_embs = []
        for search_images in all_searches:
            search_embs = []
            for search_image in search_images:
                if detect_red:
                    search_crop = face_detector.face_detect(search_image)
                    assert len(search_crop)==1
                    search_crop = search_crop[0].convert("RGB")
                else:
                    search_crop = Image.open(search_image).convert("RGB")
                
                img_cropped = preprocess(search_crop).to(device)
                
                embs = resnet(img_cropped.unsqueeze(0))
                search_embs.append(embs)
            
            search_embs = torch.cat(search_embs)
            all_searches_embs.append(search_embs)


        if verbose:
            for_loop = tqdm(os.listdir(images_folder))
        else:
            for_loop = os.listdir(images_folder)
        for person in for_loop:

            person_tensors = []
            if os.path.isfile(f"{images_folder}/{person}"):
                person_images = [f"{images_folder}/{person}"]
            else:
                person_images = [f"{images_folder}/{person}/{image_name}" for image_name in os.listdir(f"{images_folder}/{person}")]
            
            for image_path in person_images:

                crops = face_detector.face_detect(image_path)

                for crop in crops:

                    img = crop.convert("RGB")
                    
                    img_cropped = preprocess(img)
                    person_tensors.append(img_cropped)

            try:
                person_tensors = torch.stack(person_tensors).to(device)
                person_embedding = resnet(person_tensors)
                scores = [agg_func(metric_func(person_embedding.unsqueeze(0), search_embs.unsqueeze(1))) for search_embs in all_searches_embs]
            except:
                continue
            all_scores.append(scores)
            persons.append(person)

    all_scores = torch.Tensor(all_scores) # [n,m]
    n_scores = -all_scores
    n_scores -= n_scores.min(0, keepdim=True)[0]
    n_scores /= n_scores.max(0, keepdim=True)[0]
    #probs = n_scores/n_scores.sum(0, keepdim=True)
    #entropy = entropy_func(probs)
    n_scores *= 100
    all_matched_scores, all_matched_indices = n_scores.topk(topk, dim=0) # [k,m]
    matched_persons_dict = {}
    for i, search_person in enumerate(search_path):
        matched_scores = all_matched_scores[:,i].tolist()
        matched_indices = all_matched_indices[:,i].tolist()
        matched_persons = [persons[matched_ind] for matched_ind in matched_indices]
        
        matched_persons_dict[search_person] = [matched_persons, matched_scores]

    return matched_persons_dict

if __name__ == "__main__":
    images_folder = "/home/ubuntu/Nadav/data/blue" 
    #search_path = "/home/ubuntu/red/imgs/noa_3.jpg"
    search_path = "/home/ubuntu/Nadav/data/red/bar_kupershtein"
    search_path = [f"/home/ubuntu/Nadav/data/red/{search_dir}" for search_dir in os.listdir("/home/ubuntu/Nadav/data/red")]
    topk = 5
    detect_red = False
    verbose = True
    device = "cuda"

    if torch.cuda.is_available() and device=="cuda":
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    
    print(f"working with {device}")

    matched_persons_dict = search_image_in_folder(images_folder=images_folder, 
                                                            search_path=search_path, 
                                                            topk=topk, 
                                                            detect_red=detect_red, 
                                                            device=device, 
                                                            verbose=verbose)
    for search_person, matched_result in matched_persons_dict.items():
        [matched_persons, matched_scores] = matched_result
        print(search_person)
        print("")
        print(matched_persons)
        print(matched_scores)
        print("")