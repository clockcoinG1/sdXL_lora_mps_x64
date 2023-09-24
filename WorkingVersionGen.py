#!/Users/clockcoin/anaconda3/bin/python3.11
import math, os, argparse, datetime, time, logging
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, loaders, DPMSolverMultistepScheduler
from diffusers.utils.pil_utils import make_image_grid

import torch
import accelerate 
from xattr import xattr

class ImageSaver:
    def __init__(self, images, prompt, lora_model, dimensions):
        self.images = images
        self.prompt = prompt
        self.lora_model = lora_model
        self.dimensions = dimensions

    def __enter__(self):
        self.now = datetime.datetime.now()
        self.timestamp = self.now.strftime('%Y%m%d%H%M%S')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def save(self):
        for i, image in enumerate(self.images):
            unique_name = f"{self.lora_model}_{self.dimensions}{self.timestamp}_{i}.png"
            image.save(f"{unique_name}_{self.timestamp}_{self.lora_model}_{self.dimensions}.png")
            self._flash_text(f"Image saved as {unique_name}")
            logging.info(f'Image saved: {unique_name}')

    @staticmethod
    def _flash_text(text, red=False):
        color_code = "\033[41m\033[37m" if red else "\033[32m"
        print(f"{color_code}{text}\033[0m", end="\r")
        time.sleep(0.5)
        print("\033[0m", end="\r")
        time.sleep(0.5)
        logging.info(f'Flashing text printed: {text}')

def save_images(images, prompt, lora_model, dimensions):
    with ImageSaver(images, prompt, lora_model, dimensions) as saver:
        saver.save()

def generate_images(pipe, width, height, conditioning, nconditioning, pooled, npooled, num_images, num_steps):
    return [
        pipe(
            width=width, 
            height=height, 
            guidance_scale=7.0,
            negative_prompt_embeds=nconditioning,
            negative_pooled_prompt_embeds=npooled,
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            generator=torch.manual_seed(int(time.time())),
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps).images
        for _ in range(num_images)
    ]

def main(num_images, prompt, nprompt, num_steps, width, height, lora_model, show, pipe, compel):
    accelerate.init_on_device(device='mps')
    
    pipe.load_lora_weights("/Users/clockcoin/Downloads/", weight_name=lora_model)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
    )

    conditioning, pooled = compel(prompt) 
    nconditioning, npooled = compel(nprompt)
    [conditioning, nconditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, nconditioning])

    pipe.feature_extractor = None
    pipe.safety_checker = None

    images = generate_images(pipe, width, height, conditioning, nconditioning, pooled, npooled, num_images, num_steps)
    dimensions = f"{height}x{width}_{num_steps}.steps"
    save_images(images, prompt, lora_model, dimensions)
    
    if show:
        num_images = len(images)
        sqrt_num_images = math.sqrt(num_images)
        cols = math.ceil(sqrt_num_images)
        rows = math.floor(sqrt_num_images)
        if rows * cols < num_images:
            rows += 1
        make_image_grid(images=images, rows=rows , cols=cols).show(title=prompt)
 
    del pipe
    torch.mps.empty_cache()

if __name__ == "__main__":
    negative_prompt = "(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), ugly, deformed, (extra arms:1.2)"
    prompt="A (german blonde goddess) naked and aroused girl goes crazy. The woman spreads the legs as she comes towards you in full naked"

    try:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            safety_checker=None,
            use_safetensors=True,
        ).to("mps")

        compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--nprompt", type=str, default=negative_prompt)
    parser.add_argument("--lora_model", type=str, default="amateuranal.safetensors")
    parser.add_argument("--show", type=bool, default=False)
    args = parser.parse_args()
        main(args.num_images, args.prompt, args.nprompt, args.num_steps, args.width, args.height, args.lora_model, args.show, pipe, compel)
    except argparse.ArgumentError:
        parser.print_help()