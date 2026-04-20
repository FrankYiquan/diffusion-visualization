import torch

from config import *
from .prompts import load_random_prompts
from .generate import load_pipeline, generate_with_trajectory
from utils.image_utils import save_image, save_trajectory
from .visualize import plot_trajectory


def main():
    print("Using GPU:", torch.cuda.is_available())

    pipe = load_pipeline(MODEL_ID, DEVICE)
    prompts = load_random_prompts(NUM_PROMPTS, SEED)

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")

        for scale in GUIDANCE_SCALES:
            print(f"  → CFG Scale: {scale}")

            final_img, trajectory = generate_with_trajectory(
                pipe, prompt, scale
            )

            base_name = f"{OUTPUT_DIR}/prompt_{i}_scale_{scale}"

            # Save final image
            save_image(final_img, f"{base_name}_final.png")

            # Save individual steps
            save_trajectory(trajectory, base_name)

            # Save grid visualization (for report)
            plot_trajectory(
                trajectory,
                title=f"Prompt {i} | CFG={scale}",
                save_path=f"{base_name}_grid.png"
            )


if __name__ == "__main__":
    main()