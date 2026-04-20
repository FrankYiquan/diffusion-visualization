import os
import time
import torch

from config import (
    OUTPUT_DIR_2,
    SCHEDULERS,
    STEP_LIST,
    SEED,
    GUIDANCE_SCALE,
    MODEL_ID,
    DEVICE
)

from task1.generate import load_pipeline, set_scheduler


def run_experiment(prompts):
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    results = [] 
    pipe = load_pipeline(MODEL_ID, DEVICE)

    for scheduler_name in SCHEDULERS:
        print(f"\n=== Scheduler: {scheduler_name} ===")
        pipe = set_scheduler(pipe, scheduler_name)

        for steps in STEP_LIST:
            print(f"--- Steps: {steps} ---")

            for i, prompt in enumerate(prompts):

                generator = torch.Generator(device=DEVICE).manual_seed(SEED)

                start = time.time()

                image = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator
                ).images[0]

                runtime = time.time() - start

                save_path = f"{OUTPUT_DIR_2}/p{i}_{scheduler_name}_{steps}.png"
                image.save(save_path)

                print(f"p{i} | {scheduler_name} | steps={steps} | {runtime:.2f}s")

                results.append({
                    "prompt_id": i,
                    "scheduler": scheduler_name,
                    "steps": steps,
                    "runtime": runtime
                })

    return results

if __name__ == "__main__":
    from task1.prompts import load_random_prompts

    prompts = load_random_prompts(5, seed=SEED)

    results = run_experiment(prompts)