import numpy as np
import random

import torchvision.transforms.functional as TF

from PIL import Image, ImageDraw
import matplotlib.pylab as plt



def show_img_label(img,label,w_h=(50,50),thickness=2):   
    w,h=w_h                   
    cx,cy=label
    
    # draw a rectangle 
    draw = ImageDraw.Draw(img)
    draw.rectangle(((cx-w/2, cy-h/2), (cx+w/2, cy+h/2)),outline="green",width=thickness)

    plt.imshow(np.asarray(img))

def resize_images(image,label=(0.,0.), target_size=(256,256)):
    # To resize input image   
    w_origin,h_origin=image.size
    w_target,h_target=target_size
    cx,cy=label

    image_new=TF.resize(image,target_size)
    label_new = np.array([
        cx / w_origin * w_target,
        cy / h_origin * h_target
    ], dtype=np.float32)

    return image_new,label_new

def random_hflip(image,label):
    w,h=image.size
    x,y=label

    image=TF.hflip(image)
    label=w-x,y

    return image,label

def random_vflip(image,label):
    w,h=image.size
    x,y=label

    image = TF.vflip(image)
    label = x, h-y
    return image, label

def random_shift(image,label,max_translate=(0.2,0.2)):
    w,h=image.size
    max_t_w,max_t_h=max_translate
    cx,cy=label

    trans_coef=np.random.rand()*2-1
    w_t=int(trans_coef*max_t_w*w)
    h_t=int(trans_coef*max_t_h*h)

    image=TF.affine(image,translate=(w_t,h_t),shear=0,angle=0,scale=1)
    label=cx+w_t,cy+h_t

    return image,label

def transformer(image,label,params):
    
    image,label=resize_images(image,label,params["target_size"])
    if random.random() < params["p_hflip"]:
        image,label=random_hflip(image,label)
    if random.random() < params["p_vflip"]:
        image,label=random_vflip(image,label)
    if random.random() < params["p_shift"]:
        image,label=random_shift(image,label,
                                 
    params["max_translate"])
    image=TF.to_tensor(image)
    return image, label

def scale_label(a,b):
    div = [ai/bi for ai,bi in zip(a,b)]
    return div

def transformer_more(image, label, params):
    image,label=resize_images(image,label,params["target_size"])

    if random.random() < params["p_hflip"]:
        image,label=random_hflip(image,label)
        
    if random.random() < params["p_vflip"]:            
        image,label=random_vflip(image,label)
        
    if random.random() < params["p_shift"]:                            
        image,label=random_shift(image,label, params["max_translate"])

    if random.random() < params["p_brightness"]:
        brightness_factor=1+(np.random.rand()*2-1)*params["brightness_factor"]
        image=TF.adjust_brightness(image,brightness_factor)

    if random.random() < params["p_contrast"]:
        contrast_factor=1+(np.random.rand()*2-1)*params["contrast_factor"]
        image=TF.adjust_contrast(image,contrast_factor)

    if random.random() < params["p_gamma"]:
        gamma=1+(np.random.rand()*2-1)*params["gamma"]
        image=TF.adjust_gamma(image,gamma)

    if params["scale_label"]:
        label=scale_label(label,params["target_size"])
        
    image=TF.to_tensor(image)
    return image, label