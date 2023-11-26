import numpy as np
import time
from PIL import Image
from torchvision import transforms
from utils import preprocess_image, calculate_distance, retrieve_recipe, retrieve_image, retrieve_with_confidence_scores
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from data_loader import FoodDataset
import torch

def main():
    print("Initializing models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder = ImageEncoder().to(device)
    image_encoder.load_state_dict(torch.load('image_encoder.pth'))
    image_encoder.eval()

    text_encoder = TextEncoder().to(device)
    text_encoder.load_state_dict(torch.load('text_encoder.pth'))
    text_encoder.eval()

    print("Models loaded.")
    
    dist_metric = "linear"
    csv_file = '/zhome/95/b/147257/Desktop/food_bois/data/archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    img_dir = '/zhome/95/b/147257/Desktop/food_bois/data/archive/Food Images/Food Images'
    dataset = FoodDataset(csv_file=csv_file, img_dir=img_dir, transform=None)

    print("Dataset loaded.")

    query_image_path = 'sauteed-savoy-cabbage-with-scallions-and-garlic-357729.jpg'
    start_time = time.time()
    query_image = preprocess_image(query_image_path).to(device)
    image_embedding = image_encoder(query_image).detach().cpu().numpy()
    print(f"Image processed in {time.time() - start_time} seconds.")

    start_time = time.time()
    recipe_info = retrieve_recipe(image_embedding, dataset, text_encoder, device, dist_metric)
    print(f"Recipe retrieval took {time.time() - start_time} seconds.")

    # Print retrieved recipe
    if recipe_info:
        print("Retrieved Recipe:")
        # Adjust the following lines according to the actual keys
        #print(f"Title: {recipe_info['text']}")
        print(f"Input Image: {query_image_path}")
        print(f"Image Name: {recipe_info['image_name']}.jpg")
    else:
        print("Recipe not found.")




if __name__ == "__main__":
    main()
