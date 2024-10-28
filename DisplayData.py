from Dataloader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from annotations import add_weak


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

def display_drive_data():
    drive_dataset = DRIVE(transform=transform)
    drive_loader = torch.utils.data.DataLoader(drive_dataset, batch_size=1, shuffle=True)

    for index, (images, masks) in enumerate(drive_loader):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(images[0].permute(1, 2, 0))  # Convert from CHW to HWC
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(masks[0].squeeze(), cmap="gray")
        plt.axis("off")

        plt.savefig(f"drive_sample_{str(index)}.png")
        break


def display_ph2_data(display_annotations = False):
    ph2_dataset = PH2(train=True, transform=transform)
    ph2_loader = torch.utils.data.DataLoader(ph2_dataset, batch_size=1, shuffle=True)
    num_positive = 10
    num_negative = 10

    for index, (images, lesions) in enumerate(
        DataLoader(ph2_dataset, batch_size=1, shuffle=True)
    ):

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(images[0].permute(1, 2, 0))  # Convert from CHW to HWC
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Lesion Mask")
        plt.imshow(lesions[0].squeeze(), cmap="gray")
        plt.axis("off")

        if display_annotations:
            annotated_image = add_weak(images, lesions, num_positive, num_negative)
            plt.subplot(1, 3, 2)
            plt.title("Annotated Image")
            plt.imshow(annotated_image)
            plt.axis("off")

        plt.savefig(f"ph2_sample_{str(index)}.png")
        break  # Remove this to load more images

display_drive_data()
display_ph2_data(display_annotations=True)

print(f"\nThe dataset DRIVE has: {len(DRIVE(transform=transform))} images")
print(f"The dataset PH2 has: {len(PH2(train=True, transform=transform))} images\n")