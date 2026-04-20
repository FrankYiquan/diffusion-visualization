import os

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def save_trajectory(images, base_path):
    for step, img in images:
        img.save(f"{base_path}_step_{step}.png")