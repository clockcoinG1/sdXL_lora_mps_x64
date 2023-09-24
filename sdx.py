#!/Users/clockcoin/anaconda3/bin/python3.11
from abc import ABC, abstractmethod
import math, os, argparse, datetime, time, logging
from compel import Compel, ReturnedEmbeddingsType
from diffusers import  DiffusionPipeline , loaders, DPMSolverMultistepScheduler
from diffusers.utils.pil_utils import make_image_grid

import torch
import accelerate 
import os
from xattr import xattr

class ImageSaver(ABC):
    @abstractmethod
    def save(self, images, prompt, lora_model, dimensions):
        pass

class PNGImageSaver(ImageSaver):
    def __init__(self, images, prompt, lora_model, dimensions):
        self.images = images
        self.prompt = prompt
        self.lora_model = lora_model
        self.dimensions = dimensions

    def __enter__(self):
        self.now = datetime.datetime.now()
        self.timestamp = self.now.strftime('%Y%m%d%H%M%S')
        return self
    def save(self, images):
        # implementation for saving PNG images
        for i, image in enumerate(images):
            unique_name = f"image_{i}.png"
            image.save(unique_name)
            print(f"Image saved as {unique_name}")

class TextFlasher(ABC):
    @abstractmethod
    def flash_text(self, text):
        pass

class ConsoleTextFlasher(TextFlasher):
    def flash_text(self, text):
        # implementation for flashing text on console
        print(text)


class PipeClass:
    def __init__(self, lora_model):
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            safety_checker=None,
            use_safetensors=True,
        ).to("mps")
        accelerate.init_on_device(device='mps')
        self.pipe.load_lora_weights("/Users/clockcoin/Downloads/", weight_name=lora_model)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
        )
    
class ConditioningGenerator:
    def __init__(self, pipe):
        self.compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    def generate_conditioning(self, prompt, nprompt):
        conditioning, pooled = self.compel(prompt) 
        nconditioning, npooled = self.compel(nprompt)
        [conditioning, nconditioning] = self.compel.pad_conditioning_tensors_to_same_length([conditioning, nconditioning])
        return conditioning, pooled, nconditioning, npooled

class ImageGenerator:
    def __init__(self, pipe, width, height, num_images, num_steps, prompt, nprompt, conditioning_generator):
        self.pipe = pipe
        self.width = width
        self.height = height
        self.num_images = num_images
        self.num_steps = num_steps
        self.guidance_scale=7.0
        self.conditioning_generator = conditioning_generator

    def generate_images(self, prompt, nprompt):
        conditioning, pooled, nconditioning, npooled = self.conditioning_generator.generate_conditioning(prompt, nprompt)
        images = []
        for _ in range(self.num_images):
            image = pipe(
                width=self.width, 
                height=self.height, 
                guidance_scale=self.guidance_scale,
                negative_prompt_embeds=nconditioning,
                negative_pooled_prompt_embeds=npooled,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                num_images_per_prompt=self.num_images,
                num_inference_steps=self.num_steps,
                generator=torch.manual_seed(int(time.time()))
            ).images
            images.extend(image)
        return images

class Application:
    def __init__(self, image_saver, text_flasher, image_generator):
        self.image_saver = image_saver
        self.text_flasher = text_flasher
        self.image_generator = image_generator

    def run(self, prompt, nprompt):
        images = self.image_generator.generate_images(prompt, nprompt)
        self.image_saver.save(images)
        self.text_flasher.flash_text("Images saved successfully")


lora_model= "nsfw-sdxl.safetensors"
pipe = PipeClass(lora_model).pipe
variables_dict = { "width": 1024, "height": 1024, "num_steps": 25, "num_images": 1, "prompt": "A (german blonde goddess) naked and aroused girl goes crazy. The woman spreads the legs as she comes towards you in full naked", "nprompt": "(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), ugly, deformed, (extra arms:1.2)", "conditioning_generator" : ConditioningGenerator(pipe) }
image_generator = ImageGenerator(pipe, **variables_dict)
dimensions= f'{variables_dict["height"]}x{variables_dict["width"]}_{variables_dict["num_steps"]}'

image_saver = PNGImageSaver(prompt=variables_dict["prompt"], lora_model=variables_dict["lora_model"], dimensions=dimensions)
text_flasher = ConsoleTextFlasher()

app = Application(image_saver, text_flasher, image_generator)
app.run(prompt=variables_dict.get("prompt"), nprompt=variables_dict.get("nprompt"))