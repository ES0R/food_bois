# main.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Custom modules
from data_loader import FoodDataset
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from loss import ContrastiveLoss
from torch.profiler import profile, record_function, ProfilerActivity



if __name__ == '__main__':
    # Initialize data transforms, dataset, and dataloader
    transform = transforms.Compose([
        transforms.Resize((16, 16)),  # Reduced image resolution
        transforms.ToTensor(),
    ])

    csv_file = 'C:/Users/limez/OneDrive/Documents/DTU_git/food_bois/data/archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    img_dir = 'C:/Users/limez/OneDrive/Documents/DTU_git/food_bois/data/archive/Food Images/Food Images'

    dataset = FoodDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    #CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Initialize encoders, loss function, and optimizer
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    loss_fn = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)

    # Training loop with epochs
    epochs = 10
    for epoch in range(epochs):
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                images = batch['image'].to(device)
                texts = batch['text']

                # Ensure texts is processed correctly
                if isinstance(texts, str):
                    texts = [texts]

                # Prepare inputs and move to device
                inputs = text_encoder.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Start recording with profiler
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                             record_shapes=True) as prof:
                    with record_function("model_forward"):
                        # Forward pass through the encoders
                        anchor_features = image_encoder(images)
                        positive_features = text_encoder.model(**inputs).last_hidden_state.mean(dim=1)  # Use the mean of the last hidden state

                    with record_function("generate_negative_samples"):
                        # Sample negatives for contrastive loss
                        negative_indices = torch.randperm(len(dataset))[:len(images)].tolist()
                        negative_texts = [dataset[i]['text'] for i in negative_indices]
                        negative_inputs = text_encoder.tokenizer(negative_texts, return_tensors='pt', padding=True, truncation=True)
                        negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}
                        negative_features = text_encoder.model(**negative_inputs).last_hidden_state.mean(dim=1)

                    with record_function("loss_calculation"):
                        # Compute loss
                        loss = loss_fn(anchor_features, positive_features, negative_features)

                    with record_function("backpropagation"):
                        # Backpropagate and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # End of batch - print profiler results
                #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                # Update tqdm postfix
                tepoch.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1} completed")
            torch.cuda.empty_cache()  # Clear cache after each epoch

    # Save model checkpoints
    torch.save(image_encoder.state_dict(), 'image_encoder.pth')
    torch.save(text_encoder.state_dict(), 'text_encoder.pth')