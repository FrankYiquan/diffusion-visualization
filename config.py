DEVICE = "cuda"  

MODEL_ID = "runwayml/stable-diffusion-v1-5"

STEP_LIST = [20, 50, 100]
SCHEDULERS = ["ddim", "euler"]

GUIDANCE_SCALES = [5.0, 12.0]
GUIDANCE_SCALE = 7.5

NUM_PROMPTS = 5
SEED = 42
NUM_INFERENCE_STEPS = 50

SAVE_EVERY_N_STEPS = 5

OUTPUT_DIR_1 = "outputs_task1"
OUTPUT_DIR_2 = "outputs_task2"