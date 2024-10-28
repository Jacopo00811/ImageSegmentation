from PIL import ImageDraw, Image
import numpy as np

def add_weak(image, lesion, num_positive, num_negative, dot_radius=5):
    image_np = image[0].permute(1, 2, 0).numpy()
    lesion_np = lesion[0].squeeze().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image_pil)

    # Find white and black pixels in the lesion mask
    lesion_coords = np.where(lesion_np == 1)
    background_coords = np.where(lesion_np == 0)

    pos_points = []
    neg_points = []

    # Add positive labels
    if lesion_coords[0].size > 0:
        for _ in range(num_positive):
            idx = np.random.randint(0, lesion_coords[0].size)
            x = lesion_coords[1][idx]
            y = lesion_coords[0][idx]
            draw.ellipse([(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)], fill=(0, 255, 0))
            pos_points.append((x, y))

    # Add negative labels
    if background_coords[0].size > 0:
        for _ in range(num_negative):
            idx = np.random.randint(0, background_coords[0].size)
            x = background_coords[1][idx]
            y = background_coords[0][idx]
            draw.ellipse([(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)], fill=(255, 0, 0))
            neg_points.append((x, y))

    return image_pil, pos_points, neg_points