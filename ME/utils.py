import numpy as np
from data_loader import FoodDataset
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from loss import ContrastiveLoss
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.distance import cosine
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path):
    # Define the standard transformations
    transform = transforms.Compose([
        transforms.Resize((274, 169)),  # Reduced image resolution
        transforms.ToTensor(),
    ])

    # Open the image, apply the transformations and add a batch dimension
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def calculate_distance(embedding1, embedding2, dist_metric):
    if dist_metric == "linear":
        return np.linalg.norm(embedding1 - embedding2)
    elif dist_metric == "cosine":
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        return cosine(embedding1, embedding2)
    else:
        raise ValueError("Unsupported distance metric")



def retrieve_image(text_embedding, dataset, dist_metric="linear"):
    closest_image = None
    min_distance = float('inf')
    for image in dataset:
        image_embedding = get_image_embedding(image)
        distance = calculate_distance(text_embedding, image_embedding, dist_metric)
        if distance < min_distance:
            min_distance = distance
            closest_image = image
    return closest_image

def retrieve_recipe(image_embedding, dataset, text_encoder, device, dist_metric="linear"):
    closest_recipe = None
    min_distance = float('inf')
    for recipe in dataset:
        recipe_embedding = get_recipe_embedding(recipe, text_encoder, device)
        distance = calculate_distance(image_embedding, recipe_embedding, dist_metric)
        if distance < min_distance:
            min_distance = distance
            closest_recipe = recipe
    return closest_recipe

def get_recipe_embedding(recipe, text_encoder, device):
    text_data = recipe['text']
    inputs = text_encoder.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_embedding = text_encoder(inputs).detach().cpu().numpy()  # Change here
    return text_embedding


def get_image_embedding(image_path):
    image = preprocess_image(image_path).to(device)
    image_embedding = image_encoder(image).detach().cpu().numpy()
    return image_embedding

def retrieve_with_confidence_scores(embedding, dataset, encoder, device, dist_metric="linear", top_k=5):
    distances = []
    items = []

    for item in dataset:
        item_embedding = get_embedding(item, encoder, device)
        distance = calculate_distance(embedding, item_embedding, dist_metric)
        distances.append(distance)
        items.append(item)

    # Convert distances to confidence scores
    scores = np.exp(-np.array(distances))
    confidence_scores = scores / np.sum(scores)
    top_indices = np.argsort(confidence_scores)[-top_k:][::-1]
    
    top_items_with_scores = [(items[i], confidence_scores[i]) for i in top_indices]
    return top_items_with_scores


def get_embedding(item, encoder, device):
    if 'text' in item:
        # Text item
        text_data = item['text']
        inputs = encoder.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embedding = encoder(inputs).detach().cpu().numpy()  # Corrected line
    else:
        # Image item
        image = preprocess_image(item).to(device)
        embedding = encoder(image).detach().cpu().numpy()
    
    return embedding
