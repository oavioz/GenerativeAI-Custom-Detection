import extract_images


# -*- coding: utf-8 -*-
"""Open-Vocabulary Image-Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I6xAXuPPDu1_sLgVfeOb9o7l-D7UCQFx
"""

'''
Two types of encoders:  (1) for an image containing only the identification mark (e.g. tattoo)
                        (2) for general images - each patch will be encoded (possible aggregation/filtering using different model)

For the first type, we can just use CLIP as it is expected to be a crop of the image holding only the relevant data
Forst the second type, we can use DenseCLIP (with or without region proposals)
'''
import torch
import clip

from torchvision.transforms import Resize, InterpolationMode
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def get_id_encoder(backbone, device):
    model, preprocess = clip.load(backbone, device=device)
    return preprocess, model


IM_SIZE = 448 #best resolution for RN50x64

class Dense_Attention_2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.spacial_dim = spacial_dim
        self.embed_dim = embed_dim
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def interpolate_embedding(self, x):
        pe = self.positional_embedding[1:, :]
        return F.interpolate(pe.permute(1,0).reshape(1, self.embed_dim, self.spacial_dim, self.spacial_dim).to(torch.float32),
                             (x.size(2), x.size(3)))  # (HW+1)NC
    def forward(self, x):
        x += self.interpolate_embedding(x)
        x = x.to(self.v_proj.weight.dtype)

        input_dim = self.v_proj.weight.shape[1]
        mid_dim = self.v_proj.weight.shape[0]
        value = F.conv2d(x, self.v_proj.weight.view(mid_dim,input_dim,1,1), bias=self.v_proj.bias)
        output_dim = self.c_proj.weight.shape[0]
        y = F.conv2d(value, self.c_proj.weight.view(output_dim,mid_dim,1,1), bias=self.c_proj.bias)
        return y.reshape(y.shape[0], y.shape[1], y.shape[2] * y.shape[3]).permute([0,2,1])


def build_denseclip_model(backbone, device):
    # return denseCLIP encoder + preprocessing
    model, _ = clip.load(backbone, device=device)
    embed_dim = model.visual.attnpool.c_proj.in_features
    new_attnpool = Dense_Attention_2d(model.visual.input_resolution // 32, embed_dim,
                                      model.visual.attnpool.num_heads, model.visual.output_dim).to(device)
    new_attnpool.load_state_dict(model.visual.attnpool.state_dict())
    model.visual.attnpool = new_attnpool
    model.patch_size = 32
    return model

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def denseclip_transform(n_px):
    return Compose([
        Resize(size=(n_px, n_px), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_im_encoder(backbone, device):
    model = build_denseclip_model(backbone, device)
    preprocess = denseclip_transform(IM_SIZE)
    return preprocess, model

from typing import Any
import torch
import os
import tqdm
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
import clip

def custom_collate_fn(batch):
    # Separate the data samples into tensors and strings
    tensors, strings = zip(*batch)

    # Convert the list of tensors into a batch tensor
    tensors = torch.stack(tensors, dim=0)

    return tensors, strings

class ImageOnlyDataset(VisionDataset):
    def __init__(self, root, transforms):
        super(ImageOnlyDataset, self).__init__(root, transforms=transforms)
        self.imgs = extract_images.find_files(root) #All files in subdirectories
        self.imgs = [img for img in self.imgs if img.split(".")[-1] in ["jpeg", "jpg", "png"]]
        self.imgs = sorted(self.imgs)

    def _load_image(self, im_path):
        im = Image.open(im_path)
        return im

    def __getitem__(self, index: int) -> Any:
        im_path = self.imgs[index]
        im = self._load_image(im_path)
        if self.transforms:
            im = self.transforms(im)

        return im, im_path

    def __len__(self) -> int:
        return len(self.imgs)

def _save_im_enc(enc, enc_path):
    torch.save(enc, enc_path)

def _parse_im_path(im_path):
    im_name = im_path.split("/")[-1]
    return im_name

def _get_im_enc_path(enc_dir, im_name):
    return os.path.join(enc_dir, im_name + ".pt")

@torch.no_grad()    
def encode_and_save_imgs(encs_dir, ds, model, batch_size, device):
    os.makedirs(encs_dir, exist_ok=True)
    print("Encoding images:")
    dl = DataLoader(dataset=ds, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    for ims, im_paths in tqdm.tqdm(dl):
        ims = ims.to(device)
        im_encs = model.encode_image(ims)
        im_names = [_parse_im_path(im_path) for im_path in im_paths]
        enc_paths = [_get_im_enc_path(encs_dir, im_name) for im_name in im_names]
        for im_enc, enc_path in zip(im_encs, enc_paths):
            _save_im_enc(im_enc, enc_path)
    print("Done, encodings are located at {}".format(encs_dir))


import glob

def _load_im_enc(enc_path):
    return torch.load(enc_path)


OWL_VIT_CLIP_BEST_PROMPT_TEMPLATES = [
    'itap of a {}.',
    'a bad photo of the {}.',
    'a origami {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
]

def _basic_formatting(category, template):
    return template.format(category)

def build_text_embedding(text, model, device):
    texts = []
    for template in OWL_VIT_CLIP_BEST_PROMPT_TEMPLATES:
            texts.append(_basic_formatting(text, template))
    texts = clip.tokenize(texts)
    texts_emb = model.encode_text(texts.to(device))
    texts_emb = torch.mean(texts_emb, dim=0, keepdim=True)
    return texts_emb

@torch.no_grad()
def _order_images_by_similarity(encs_dir, preprocess, model, device, txt):
    ref_enc = build_text_embedding(txt, model, device)
    # should be 1 x EMBEDDING_SIZE
    ref_enc = ref_enc / ref_enc.norm()
    scores = []
    encs_paths = glob.glob(encs_dir + '/*.pt')
    print("Ordering images by similarity...")
    for enc_path in tqdm.tqdm(encs_paths):
        enc = _load_im_enc(enc_path)
        # Should be n X EMBEDDINGS_SIZE
        enc = enc / torch.linalg.norm(enc, dim=-1, keepdim=True)
        sim = torch.einsum('ac,bc->b', ref_enc, enc.to(ref_enc.dtype))
        max_sim = torch.max(sim)
        scores.append((max_sim.cpu().numpy(), enc_path.split("/")[-1][:-3]))
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    print("Done")
    return scores

def search_image(encs_dir, preprocess, model, output_path, device, txt):
    ims_paths_by_sim = _order_images_by_similarity(encs_dir, preprocess, model, device, txt)

    print("Saving results to {}".format(output_path))
    with open(output_path, 'w') as wf:
        for score, im_name in ims_paths_by_sim:
            wf.write("{}; {}".format(score, im_name))

    return ims_paths_by_sim

import argparse
import os
import glob
from PIL import Image

def build_db(args):
    preprocess, model = get_im_encoder(args.backbone, args.device)
    ds = ImageOnlyDataset(args.im_dir, preprocess)
    encode_and_save_imgs(encs_dir=args.enc_dir,ds=ds, model=model, batch_size=args.batch_size, device=args.device)

def search_db(args):
    preprocess, model = get_id_encoder(backbone=args.backbone, device=args.device)

    search_result = search_image(encs_dir=args.enc_dir, preprocess=preprocess,
                                 model=model,
                                 output_path=args.output_path,
                                 device=args.device, 
                                 txt=args.txt)
    if args.im_dir is None:
        exit(0)

    for idx, sample in enumerate(search_result):
        if idx == 10:
            break
        score, im_name = sample
        im_path = os.path.join(args.im_dir, im_name)
        im = Image.open(im_path)
        im.show(title="{}:{}".format(idx, score))

cmd_cfg = {
  'im_dir' : '../tests_and_examples/documents/Sunflower/Sunflower_Downy mildew/',
  'enc_dir' : './encs/',
  'device' : "cuda" if torch.cuda.is_available() else "cpu", 
  'backbone' : "RN50x64",
  'batch_size' : 4,
  'ref_im_path' : None,
  'output_path' : "results.txt",
  'txt' : "Sunflower"
}

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dict_to_class(d):
  return Struct(**d)

CMD = "search" #"search_db"
if CMD == "build_db":
  build_db(dict_to_class(cmd_cfg))
else:
  search_db(dict_to_class(cmd_cfg))