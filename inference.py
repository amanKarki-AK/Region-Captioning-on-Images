import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# ------------------
# Helper: load image
# ------------------
def load_image(image_path):
    """Loads and returns a PIL image in RGB format."""
    return Image.open(image_path).convert("RGB")

# ------------------
# Preprocessing helpers
# ------------------
def preprocess_for_detector(image_pil):
    """Prepares a PIL image for the Faster R-CNN detector."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image_pil)

def preprocess_for_generative_encoder(region_pil):
    """Prepares a cropped PIL image region for the GenerativeEncoderCNN."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    return transform(region_pil).unsqueeze(0)

def tokens_to_sentence(tokens, vocab):
    """Converts a list of token IDs back into a string."""
    caption_words = []
    for word_id in tokens:
        word = vocab.idx2word[word_id.item()]
        if word == "<end>":
            break
        if word != "<start>":
            caption_words.append(word)
    return " ".join(caption_words)

# ------------------
# Captioning Functions
# ------------------

def caption_image(image_pil, encoder, decoder, vocab, device):
    """
    Generates a caption for a *full* image.
    Used for COCO evaluation.
    """
    # Preprocess the full image for the generative encoder
    image_tensor = preprocess_for_generative_encoder(image_pil).to(device)
    
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features, vocab)
        
    return tokens_to_sentence(sampled_ids[0], vocab)


def caption_region(region_pil, encoder, decoder, vocab, device):
    """
    Generates a caption for a *cropped region* of an image.
    """
    region_tensor = preprocess_for_generative_encoder(region_pil).to(device)
    
    with torch.no_grad():
        features = encoder(region_tensor)
        sampled_ids = decoder.sample(features, vocab) # [1, seq_len]

    return tokens_to_sentence(sampled_ids[0], vocab)

# ------------------
# Main: Generate captions for Faster R-CNN regions
# ------------------
def generate_region_captions(image_path, faster_rcnn_model, encoder, decoder, vocab, device, conf_thresh=0.5):
    """
    Detect regions with Faster R-CNN, then generate captions for each region.
    Returns image with drawn boxes + captions and list of region captions.
    """
    image = load_image(image_path)
    
    # 1. Get region proposals from Faster R-CNN
    image_tensor = preprocess_for_detector(image).to(device)
    faster_rcnn_model.confidence_threshold = conf_thresh # Update threshold
    
    with torch.no_grad():
        boxes = faster_rcnn_model(image_tensor) # [N, 4]

    draw = ImageDraw.Draw(image)
    try:
        # Load a slightly better font if available, otherwise default
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        
    region_captions = []

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()

        # 2. Crop region
        region_pil = image.crop((x1, y1, x2, y2))

        # 3. Caption region
        caption = caption_region(region_pil, encoder, decoder, vocab, device)
        region_captions.append(((x1, y1, x2, y2), caption))
        
        # 4. Draw on image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw text background
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1), caption, fill="white", font=font)

    return image, region_captions

'''

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ------------------
# Helper: load image
# ------------------
def load_image(image_path):
    """Loads an image and converts it to RGB."""
    return Image.open(image_path).convert("RGB")

# ------------------
# Preprocess region for encoder
# ------------------
def preprocess_region(region_image):
    """Prepares a single cropped PIL image for the encoder."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    # Returns a 4D tensor [1, C, H, W]
    return transform(region_image).unsqueeze(0)

# ------------------
# Caption one region
# ------------------
def caption_region(region_image, encoder, decoder, vocab, device):
    """
    Generates a caption for a single cropped PIL image region.
    'region_image' is a PIL Image.
    """
    # 'region_tensor' will be a 4D tensor [1, 3, 224, 224]
    region_tensor = preprocess_region(region_image).to(device)
    
    with torch.no_grad():
        # This encoder *correctly* receives a 4D tensor
        features = encoder(region_tensor)
        sampled_ids = decoder.sample(features, vocab)

    caption_words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == "<end>":
            break
        if word != "<start>":
            caption_words.append(word)

    return " ".join(caption_words)

# ------------------
# Main: Generate captions for R-CNN regions
# ------------------
def generate_region_captions(image_path, rcnn_model, encoder, decoder, vocab, device, conf_thresh=0.5):
    """
    Detect regions with Faster R-CNN, then generate captions for each region.
    Returns image with non-overlapping boxes/captions.
    """
    image = load_image(image_path)
    
    # Pre-process the full image for Faster R-CNN
    transform_rcnn = transforms.Compose([transforms.ToTensor()])
    # image_tensor is a 4D tensor: [1, 3, H, W]
    image_tensor = transform_rcnn(image).unsqueeze(0).to(device)

    # --- FIX 1: TENSOR DIMENSION FIX ---
    # The R-CNN model, in eval() mode, expects a LIST of 3D tensors.
    # We convert our 4D batch tensor [1, 3, H, W] into a list: [[3, H, W]]
    with torch.no_grad():
        # This is the line that fixes the error.
        detections = rcnn_model([image_tensor[0]])
    # --- END FIX 1 ---
    
    # Filter detections by confidence
    boxes = detections[0]['boxes']
    scores = detections[0]['scores']
    
    keep_indices = [i for i, score in enumerate(scores) if score > conf_thresh]
    boxes_to_caption = boxes[keep_indices]

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except IOError:
        print("Default font not found, using basic PIL drawing.")
        font = None

    region_captions_list = []

    for box in boxes_to_caption:
        box_coords = [int(c) for c in box.tolist()]
        x1, y1, x2, y2 = box_coords

        # Crop region (This is a PIL Image)
        region_pil_image = image.crop((x1, y1, x2, y2))

        # Caption region
        caption = caption_region(region_pil_image, encoder, decoder, vocab, device)
        region_captions_list.append((box_coords, caption))

        # --- FIX 2: OVERLAP AND CUT-OFF FIX ---
        
        # 1. Get text size
        text_height = 10
        text_width = len(caption) * 6 # A rough estimate if font fails
        if font:
            try:
                # Get actual bounding box of the text
                text_bbox = font.getbbox(caption)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # Fallback for older PIL versions
                text_size = font.getsize(caption)
                text_width = text_size[0]
                text_height = text_size[1]

        # 2. Determine text position
        text_padding = 3
        box_padding = 2
        text_y_pos = y1 - text_height - (text_padding * 2) - box_padding # Default: above the box

        # 3. Check if text is cut off at the top
        if text_y_pos < 0:
            text_y_pos = y1 + box_padding + text_padding # Place it inside the box

        text_x_pos = x1 + box_padding

        # 4. Draw text background
        draw.rectangle(
            [text_x_pos - text_padding, 
             text_y_pos - text_padding, 
             text_x_pos + text_width + text_padding, 
             text_y_pos + text_height + text_padding],
            fill='white'
        )

        # 5. Draw the bounding box (drawn after text so text is on top)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # 6. Draw the caption text
        draw.text((text_x_pos, text_y_pos), caption, fill='red', font=font)
        # --- END FIX 2 ---

    # We return the PIL Image for main.py to save
    return image, region_captions_list

# ------------------
# Full Image Captioning (for evaluate.py)
# ------------------
def caption_image(image_pil, encoder, decoder, vocab, device):
    """
    Generates a single caption for the entire image.
    Used by evaluate.py
    """
    # Use the same preprocessing as for regions
    image_tensor = preprocess_region(image_pil).to(device)
    with torch.no_grad():
        # This encoder *correctly* receives a 4D tensor
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features, vocab)

    caption_words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == "<end>":
            break
        if word != "<start>":
            caption_words.append(word)

    return " ".join(caption_words)

'''