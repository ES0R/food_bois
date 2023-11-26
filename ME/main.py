# main.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
# Custom modules
from data_loader import FoodDataset
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from loss import ContrastiveLoss
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt 

def save_plot(data, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.savefig(filename)


if __name__ == '__main__':

    # Data loader
    transform = transforms.Compose([
        transforms.Resize((274, 169)),  # Reduced image resolution
        transforms.ToTensor(),
    ])

    csv_file = '/zhome/95/b/147257/Desktop/food_bois/data/archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    img_dir = '/zhome/95/b/147257/Desktop/food_bois/data/archive/Food Images/Food Images'

    dataset = FoodDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)


    #CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Initialize encoders, loss function, and optimizer
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    loss_fn = ContrastiveLoss(margin=0.5)
    optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)


    # Initialize logging variables
    epoch_losses = []
    epoch_times = []

    # Training loop with epochs
    epochs = 25
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total_val_loss = 0

        # Training phase
        with tqdm(train_loader, unit="batch") as train_epoch:
            for batch in train_epoch:
                train_epoch.set_description(f"Epoch {epoch + 1}")

                images = batch['image'].to(device)
                texts = batch['text']

                # Ensure texts is processed correctly
                if isinstance(texts, str):
                    texts = [texts]

                # Prepare inputs and move to device
                inputs = text_encoder.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass through the encoders
                anchor_features = image_encoder(images)
                positive_features = text_encoder.model(**inputs).last_hidden_state.mean(dim=1)

                # Sample negatives for contrastive loss
                negative_indices = torch.randperm(len(dataset))[:len(images)].tolist()
                negative_texts = [dataset[i]['text'] for i in negative_indices]
                negative_inputs = text_encoder.tokenizer(negative_texts, return_tensors='pt', padding=True, truncation=True)
                negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}
                negative_features = text_encoder.model(**negative_inputs).last_hidden_state.mean(dim=1)

                # Compute loss
                loss = loss_fn(anchor_features, positive_features, negative_features)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_epoch.set_postfix(loss=loss.item())

       
        epoch_duration = time.time() - start_time
        epoch_losses.append(total_loss / len(train_loader))
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch + 1} completed")

    # Save model checkpoints
    torch.save(image_encoder.state_dict(), 'image_encoder_new_1_25.pth')
    torch.save(text_encoder.state_dict(), 'text_encoder_new_1_25.pth')

    # Save plots
    save_plot(epoch_losses, 'Training Loss', 'Loss', 'loss_plot_new_1_25.png')
    save_plot(epoch_times, 'Time per Epoch', 'Time (s)', 'time_per_epoch_plot_new_1_25.png')










