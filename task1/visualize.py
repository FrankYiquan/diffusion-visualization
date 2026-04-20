import matplotlib.pyplot as plt

def plot_trajectory(images, title, save_path):
    n = len(images)

    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    if n == 1:
        axes = [axes]

    for i, (step, img) in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f"Step {step}")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()