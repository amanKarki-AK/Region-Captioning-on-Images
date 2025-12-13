'''import torch
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms

# Import from our other project files
from model import GenerativeEncoderCNN, DecoderMRNN, FasterRCNNRegionDetector
from vocabulary import Vocabulary
from inference import generate_region_captions

def main():
    """
    Main function to run region-level captioning inference on a test image.
    """
    # -----------------
    # Configuration
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths (update these to your model and data paths)
    vocab_path = 'Project/data/vocab.pkl'
    encoder_path = 'Project/models/gen_encoder_cnn-10.pth'
    decoder_path = 'Project/models/decoder_mrnn-10.pth'
    
    # Example test image (update this to your image)
    test_image = "Project/data/Images/1000523639.jpg" 

 
    # Model parameters
    embed_size = 512
    hidden_size = 512
    
    # -----------------
    # Load Vocabulary
    # -----------------
    print(f"Loading vocabulary from {vocab_path}...")
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded. Size: {len(vocab)}")
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}")
        print("Please run build_vocab.py first.")
        return

    # -----------------
    # Load Models
    # -----------------
    print("Loading models...")
    
    # 1. Faster R-CNN Detector
    # This model is pre-trained by torchvision
    faster_rcnn_model = FasterRCNNRegionDetector(confidence_threshold=0.5).to(device)
    faster_rcnn_model.eval()

    # 2. Generative Encoder (trained by you)
    gen_encoder_cnn = GenerativeEncoderCNN(embed_size).to(device)
    try:
        gen_encoder_cnn.load_state_dict(torch.load(encoder_path, map_location=device))
        gen_encoder_cnn.eval()
        print(f"Loaded generative encoder from {encoder_path}")
    except FileNotFoundError:
        print(f"Warning: Encoder model file not found at {encoder_path}. Using random weights.")
    except Exception as e:
        print(f"Error loading encoder model: {e}")
        return

    # 3. Generative Decoder (trained by you)
    decoder_mrnn = DecoderMRNN(embed_size, hidden_size, len(vocab)).to(device)
    try:
        decoder_mrnn.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder_mrnn.eval()
        print(f"Loaded generative decoder from {decoder_path}")
    except FileNotFoundError:
        print(f"Warning: Decoder model file not found at {decoder_path}. Using random weights.")
    except Exception as e:
        print(f"Error loading decoder model: {e}")
        return

    # -----------------
    # Run Inference
    # -----------------
    print(f"\nGenerating region captions for {test_image}...")
    
    try:
        final_image, region_caps = generate_region_captions(
            test_image,
            faster_rcnn_model, # Use Faster R-CNN
            gen_encoder_cnn,
            decoder_mrnn,
            vocab,
            device,
            conf_thresh=0.5 # Confidence for Faster R-CNN
        )
        
        print("\n--- Generated Captions ---")
        if not region_caps:
            print("No regions detected with confidence > 0.5")
        
        for box, cap in region_caps:
            print(f"Box {box}: {cap}")
        print("--------------------------")

        # Show the final annotated image
        plt.figure(figsize=(12, 12))
        plt.imshow(final_image)
        plt.title("Region Captioning with Faster R-CNN")
        plt.axis("off")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Test image not found at {test_image}")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()'''

import torch
import pickle
import os
import matplotlib.pyplot as plt # We still need this for some setup, but won't use .show()

# Import from our project files
from vocabulary import Vocabulary
from model import GenerativeEncoderCNN, DecoderMRNN, FasterRCNNRegionDetector
from inference import generate_region_captions # This function does all the work

def main():
    """
    Main function to run inference on a single test image.
    """
    
    # -----------------
    # Configuration
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- UPDATE THESE PATHS ---
    # Point these to your final trained models from Stage 2
    encoder_path = 'Project/models/gen_encoder_cnn-30.pth'
    decoder_path = 'Project/models/decoder_mrnn-30.pth'
    
    # --- UPDATE THIS IMAGE ---
    # Point this to any image you want to test
    test_image = "Project/data/test_img/Screenshot 2025-10-29 140418.png" 
 
    vocab_path = 'Project/data/vocab.pkl'
    
    # Model parameters
    embed_size = 512
    hidden_size = 512

    # -----------------
    # Load Vocabulary
    # -----------------
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # -----------------
    # Load Models
    # -----------------
    print("Loading models...")
    
    # Generative Models (Stage 2)
    encoder = GenerativeEncoderCNN(embed_size).to(device)
    decoder = DecoderMRNN(embed_size, hidden_size, len(vocab)).to(device)
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    encoder.eval() # Set to evaluation mode
    decoder.eval() # Set to evaluation mode

    # Region Detector (Faster R-CNN)
    faster_rcnn_model = FasterRCNNRegionDetector().to(device)
    faster_rcnn_model.eval() # Set to evaluation mode

    # -----------------
    # Run Inference
    # -----------------
    print(f"Running inference on image: {test_image}")
    
    # This function returns the final PIL image with boxes/text drawn on it
    # and a list of (box, caption) tuples
    final_image_with_boxes, region_caps = generate_region_captions(
        image_path=test_image,
        faster_rcnn_model=faster_rcnn_model,
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        device=device,
    )
    
    # -----------------
    # Print Captions to Terminal
    # -----------------
    print("\n--- Generated Captions ---")
    if not region_caps:
        print("No regions were detected or captioned.")
    for box, cap in region_caps:
        # Box coordinates are (x1, y1, x2, y2)
        print(f"Region at [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]: {cap}")
    print("------------------------")

    # ---------------------------------
    # --- MODIFICATION: Save the image ---
    # ---------------------------------
    
    # We will no longer try to 'show' the image with plt.show()
    # Instead, we just save the final PIL image to a file.
    
    save_path = "result_image.jpg"
    try:
        final_image_with_boxes.save(save_path)
        print(f"\nSuccessfully saved annotated image to: {save_path}")
        print("Please open this file in your file explorer to see the result.")
    except Exception as e:
        print(f"\nError saving image: {e}")
        print("Could not save the final image.")


if __name__ == "__main__":
    main()

