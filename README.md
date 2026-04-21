# Diffusion Visualization (SDXL)

## Summary

This project uses **Stable Diffusion XL (SDXL)** via Hugging Face Diffusers to visualize the reverse diffusion process and analyze how images are generated from noise.

* **Task 1:** Generates images from DrawBench prompts and visualizes intermediate denoising steps to show how structure and details emerge over time.
* **Task 2:** Compares different schedulers and sampling steps (e.g., 20, 50, 100) to study tradeoffs between speed, stability, and image quality.

## Setup & Run

To install dependencies
```bash
pip install -r requirements.txt
```
To run task 1
```bash
python -m task1.main
```
To run task 2
```bash
python -m task2.main
```

## Project Structure

```
Diffusion-Visualization/
│
├── task1/
│   ├── generate.py
│   ├── visualize.py
│   ├── prompts.py
│   └── main.py
│
├── task2/
│   └── main.py
│
├── utils/
│   └── image_utils.py
│
├── config.py
├── requirements.txt
└── README.md
```
