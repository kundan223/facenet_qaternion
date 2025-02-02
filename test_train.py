import torch
import numpy as np
from model_modified import FaceNetModel
from loss import TripletLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.distance import PairwiseDistance
import time

# Update image_size to 4 channels
batch_size = 64
image_size = (4, 224, 224)  # Now 4 channels for quaternions
num_epochs = 200
margin = 0.5
learning_rate = 0.001
step_size = 50

def generate_quaternion_data(batch_size, original_channels=3):
    """Generates 4-channel data with the 4th channel derived from RGB."""
    # Simulate RGB images (batch_size, 3, H, W)
    rgb_data = torch.randn(batch_size, original_channels, 224, 224)
    
    # Compute 4th channel (Option 1: Zero-padded)
    fourth_channel = torch.zeros_like(rgb_data[:, :1, :, :])  # Zero-padded
    
    # Option 2: Grayscale (uncomment to use)
    # grayscale = 0.299 * rgb_data[:, 0:1, :, :] + 0.587 * rgb_data[:, 1:2, :, :] + 0.114 * rgb_data[:, 2:3, :, :]
    # fourth_channel = grayscale
    
    # Concatenate along channel dimension
    quat_data = torch.cat([rgb_data, fourth_channel], dim=1)
    return quat_data

# Initialize model and move to device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = FaceNetModel(pretrained=False).to(device)
triplet_loss = TripletLoss(margin).to(device)
l2_dist = PairwiseDistance(2)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

def train(model, optimizer, scheduler, triplet_loss, num_epochs, batch_size, image_size):
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        start_time = time.time()
        model.train()
        total_loss = 0

        for _ in range(100):  # Simulate batches
            # Generate 4-channel data
            anc_img = generate_quaternion_data(batch_size).to(device)
            pos_img = generate_quaternion_data(batch_size).to(device)
            neg_img = generate_quaternion_data(batch_size).to(device)

            # Forward pass
            anc_embed = model(anc_img)
            pos_embed = model(pos_img)
            neg_embed = model(neg_img)

            # Compute loss
            pos_dist = l2_dist(anc_embed, pos_embed)
            neg_dist = l2_dist(anc_embed, neg_embed)
            loss = triplet_loss(anc_embed, pos_embed, neg_embed)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = total_loss / 100
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time for epoch: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    train(model, optimizer, scheduler, triplet_loss, num_epochs, batch_size, image_size)