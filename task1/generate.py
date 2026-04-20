import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler
from config import *

def load_pipeline(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe = pipe.to(device)

    # # debug
    # print("Tokenizer 1:", pipe.tokenizer)
    # print("Tokenizer 2:", pipe.tokenizer_2)

    return pipe

def set_scheduler(pipe, scheduler_name):
    config = pipe.scheduler.config

    if scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(config)

    elif scheduler_name == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(config)

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return pipe


def generate_with_trajectory(pipe, prompt, guidance_scale):
    intermediate_images = []

    def callback(step, timestep, latents):
        if step % SAVE_EVERY_N_STEPS == 0:
            image = pipe.decode_latents(latents)
            image = pipe.numpy_to_pil(image)[0]
            intermediate_images.append((step, image))

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=callback,
            callback_steps=1
        )

    final_image = result.images[0]
    return final_image, intermediate_images