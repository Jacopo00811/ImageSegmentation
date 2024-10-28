import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Loss import point_loss
from Dataloader import PH2  # Ensure PH2 is imported correctly
from annotations import add_weak

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # RGB input
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)  # Output for binary mask
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x

def test_with_real_data():
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    ph2_dataset = PH2(train=True, transform=None)  # Adjust transform as needed
    ph2_loader = DataLoader(ph2_dataset, batch_size=1, shuffle=True)

    num_positive = 10
    num_negative = 10

    for epoch in range(5):  # Run for a few epochs
        for idx, (images, lesions) in enumerate(ph2_loader):
            optimizer.zero_grad()
            outputs = model(images)

            # Get annotated points
            annotated_image, pos_points, neg_points = add_weak(images, lesions, num_positive, num_negative)

            # Ensure pos_points and neg_points are in the correct format
            pos_points = torch.tensor(pos_points)  # Convert to tensor
            neg_points = torch.tensor(neg_points)  # Convert to tensor

            # Check if pos_points and neg_points are not empty
            if pos_points.numel() == 0 or neg_points.numel() == 0:
                print("No positive or negative points found.")
                continue

            loss = point_loss(outputs, lesions, pos_points, neg_points)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN or Inf at epoch {epoch}, index {idx}")
                return

            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Real Loss: {loss.item()}")

# Run tests
# test_with_synthetic_data()  # Quick check with synthetic data
test_with_real_data()  # More comprehensive test with real data
