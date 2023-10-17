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

def search_image_in_folder(images_folder = "/home/opc/Nadav/data/blue", 
                           search_path = "/home/opc/red/imgs/noa_3.jpg", 
                           topk = 5, 
                           model_name = "casia-webface", 
                           metric = "l2", 
                           agg = "mean", 
                           face_detector_path = "saved_models/yolov8n-face.onnx"):
    """
    Returns topk matches of search_path in images_folder, and their scores.

    Parameters:
        images_folder (str): The folder path of images (blue or red).
        search_path (str): The image path to search (blue or red).
        topk (int): How many to return.
        model_name (str): Face recognition model: "casia-webface", "vggface2".
        metric (str): Correlation metric: "l1", "l2", "cosine".
        agg (str): Aggregation function: "min", "mean".
        face_detector_path (str): Face detection model: "saved_models/yolov8n-face.onnx".

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
    resnet = InceptionResnetV1(pretrained=model_name).eval()

    print(f"model_name: {model_name} | metric: {metric} | agg: {agg}\n")

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
    scores = []

    with torch.no_grad():

        search_crop = face_detector.face_detect(search_path)
        assert len(search_crop)==1
        search_crop = search_crop[0].convert("RGB")
        
        img_cropped = preprocess(search_crop)
        
        search_embs = resnet(img_cropped.unsqueeze(0))


        for person in tqdm(os.listdir(images_folder)):

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
                person_tensors = torch.stack(person_tensors)
                person_embedding = resnet(person_tensors)
                score = agg_func(metric_func(person_embedding.unsqueeze(0), search_embs.unsqueeze(1)))
            except:
                continue
            scores.append(score)
            persons.append(person)

    scores = torch.Tensor(scores)
    n_scores = -scores
    n_scores -= n_scores.min()
    n_scores /= n_scores.max()
    probs = n_scores/n_scores.sum()
    entropy = entropy_func(probs)
    n_scores *= 100
    match_indices = n_scores.topk(topk)[1].tolist()
    matched_persons = [persons[match_ind] for match_ind in match_indices]
    matched_scores = [n_scores[match_ind].item() for match_ind in match_indices]

    return matched_persons, matched_scores