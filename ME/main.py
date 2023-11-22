import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Custom modules
from data_loader import FoodDataset
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from loss import ContrastiveLoss

# Initialize data transforms, dataset, and dataloader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

csv_file = 'C:/Users/Emil/Documents/DTU_git/food_bois/data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
img_dir= 'C:/Users/Emil/Documents/DTU_git/food_bois/data/Food Images/Food Images'


dataset = FoodDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize image and text encoders
image_encoder = ImageEncoder()
text_encoder = TextEncoder()

# Initialize loss function and optimizer
loss_fn = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)

# Training loop
epochs = 10
# main.py
# ... [rest of your imports and initializations]

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        images = batch['image']
        texts = batch['text']

        # Ensure that texts is a list of strings
        if isinstance(texts, str):
            texts = [texts]

        # Forward pass through the encoders
        anchor_features = image_encoder(images)
        positive_features = text_encoder(texts)

        # Sample negative examples for contrastive loss
        negative_indices = torch.randperm(len(dataset))[:len(images)].tolist()
        negative_texts = [dataset[i]['text'] for i in negative_indices]
        negative_features = text_encoder(negative_texts)

        # Compute contrastive loss
        loss = loss_fn(anchor_features, positive_features, negative_features)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed")

# Save model checkpoints
torch.save(image_encoder.state_dict(), 'image_encoder.pth')
torch.save(text_encoder.state_dict(), 'text_encoder.pth')
