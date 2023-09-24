#!/Users/clockcoin/anaconda3/bin/python3.11
import os
import argparse
import datetime
import time
import logging
import logging.handlers
from abc import ABC, abstractmethod
from colorama import Fore, Style
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, loaders, DPMSolverMultistepScheduler
from diffusers.utils.pil_utils import make_image_grid
import torch
import accelerate
import coloredlogs


class Logger:
    def __init__(self, name, level=logging.DEBUG, log_file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)
        formatter = logging.Formatter(
            "\t\t%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
        )
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.console_handler)
        if log_file:
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=1000000, backupCount=5
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.error(f"Failed to create file handler: {e}")


class ImageSaver(ABC):
    @abstractmethod
    def save(self, images):
        pass


class PNGImageSaver(ImageSaver):
    def __init__(self, prompt, lora_model, dimensions, output_dir, base_path):
        self.prompt = prompt
        self.lora_model = lora_model
        self.output_dir = output_dir
        self.base_path = base_path
        self.dimensions = dimensions
        self.image_count = 0

    def save(self, images):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        for image in images:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S%f")
            unique_name = f"sdxl-{self.lora_model.split('/')[-1]}{self.dimensions}_{timestamp}_{str(self.image_count)}.png"
            unique_name = os.path.join(self.output_dir, unique_name)
            image.save(unique_name)
            logger.warning("*" * 25)
            logger.debug(
                f"Image #{str(self.image_count)} saved with as: {unique_name}\t!"
            )
            logger.debug(f"Prompt: {self.prompt[0:75]}\tSettings: {self.dimensions}\t")
            logger.debug(f"Lora Model: {self.lora_model}")
            logger.warning("*" * 25)
            self.image_count += 1


class PipeClass:
    def __init__(self, lora_model, base_path):
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            safety_checker=None,
            use_safetensors=True,
        ).to("mps")
        accelerate.init_on_device(device="mps")
        self.pipe.load_lora_weights(base_path, weight_name=lora_model)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
        )


class ConditioningGenerator:
    def __init__(self, pipe):
        self.compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

    def generate_conditioning(self, prompt, nprompt, show):
        conditioning, pooled = self.compel(prompt)
        nconditioning, npooled = self.compel(nprompt)
        [
            conditioning,
            nconditioning,
        ] = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning, nconditioning]
        )
        return conditioning, pooled, nconditioning, npooled


class ImageGenerator:
    def __init__(
        self,
        pipe,
        width,
        height,
        num_images,
        num_steps,
        prompt,
        nprompt,
        conditioning_generator,
        show,
    ):
        self.pipe = pipe
        self.width = width
        import argparse
        import os
        import time
        from typing import List

        import coloredlogs
        import torch
        from PIL import Image
        from termcolor import colored
        from torchvision.transforms import ToPILImage

        from logger import Logger


        class PipeClass:
            def __init__(self, lora_model: str, base_path: str):
                self.lora_model = lora_model
                self.base_path = base_path

            def pipe(self, **kwargs):
                pass


        class ConditioningGenerator:
            def __init__(self, pipe: PipeClass):
                self.pipe = pipe

            def generate_conditioning(self, prompt: str, nprompt: str, show: bool):
                pass


        class ImageGenerator:
            def __init__(
                self,
                pipe: PipeClass,
                width: int,
                height: int,
                num_images: int,
                num_steps: int,
                prompt: str,
                nprompt: str,
                conditioning_generator: ConditioningGenerator,
                show: bool,
            ):
                self.pipe = pipe
                self.width = width
                self.height = height
                self.num_images = num_images
                self.num_steps = num_steps
                self.prompt = prompt
                self.nprompt = nprompt
                self.conditioning_generator = conditioning_generator
                self.show = show

            def generate_images(self):
                (
                    conditioning,
                    pooled,
                    nconditioning,
                    npooled,
                ) = self.conditioning_generator.generate_conditioning(
                    self.prompt, self.nprompt, self.show
                )
                for _ in range(self.num_images):
                    image = self.pipe(
                        width=self.width,
                        height=self.height,
                        guidance_scale=7.0,
                        negative_prompt_embeds=nconditioning,
                        negative_pooled_prompt_embeds=npooled,
                        prompt_embeds=conditioning,
                        pooled_prompt_embeds=pooled,
                        num_images_per_prompt=self.num_images,
                        num_inference_steps=self.num_steps,
                        generator=torch.manual_seed(int(time.time())),
                    ).images
                    for img in image:
                        yield img


        class ImageSaver:
            def save(self, image: Image):
                pass


        class PNGImageSaver(ImageSaver):
            def __init__(
                self,
                prompt: str,
                lora_model: str,
                dimensions: str,
                output_dir: str,
                base_path: str,
            ):
                self.prompt = prompt
                self.lora_model = lora_model
                self.dimensions = dimensions
                self.output_dir = output_dir
                self.base_path = base_path

            def save(self, image: Image):
                to_pil = ToPILImage()
                image = to_pil(image[0].cpu())
                filename = f"{self.prompt}_{self.lora_model}_{self.dimensions}.png"
                filepath = os.path.join(self.output_dir, filename)
                image.save(filepath)


        class Application:
            def __init__(self, image_saver: ImageSaver, image_generator: ImageGenerator):
                self.image_saver = image_saver
                self.image_generator = image_generator

            def run(self):
                num_images_saved = 0
                for image in self.image_generator.generate_images():
                    self.image_saver.save(image)
                    num_images_saved += 1
                print(
                    colored(
                        f"{num_images_saved} images saved successfully",
                        "green",
                    )
                )


        class ArgumentParser:
            def __init__(self):
                self.parser = argparse.ArgumentParser()
                self.parser.add_argument("--base_path", type=str, default=f"{os.getcwd()}")
                self.parser.add_argument("--height", type=int, default=1024)
                self.parser.add_argument(
                    "--lora_model", type=str, default=f"{os.getcwd()}/.tensors/am.safetensors"
                )
                self.parser.add_argument(
                    "--nprompt",
                    type=str,
                    default="(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), ugly, deformed, (extra arms:1.2)",
                )
                self.parser.add_argument("--num_images", type=int, default=1)
                self.parser.add_argument("--num_steps", type=int, default=10)
                self.parser.add_argument(
                    "--output_dir", type=str, default=f"{os.getcwd()}/media/"
                )
                self.parser.add_argument("--prompt", type=str, default="....")
                self.parser.add_argument("--show", type=bool, default=False)
                self.parser.add_argument("--width", type=int, default=1024)

            def parse_arguments(self):
                args = self.parser.parse_args()
                if args.num_steps < 1:
                    raise ValueError("num_steps must be at least 1")
                if args.width < 1:
                    raise ValueError("width must be at least 1")
                if args.height < 1:
                    raise ValueError("height must be at least 1")
                if args.num_images < 1:
                    raise ValueError("num_images must be at least 1")
                return args


        def main():
            argument_parser = ArgumentParser()
            args = argument_parser.parse_arguments()
            lora_model = args.lora_model
            pipe = PipeClass(lora_model, args.base_path).pipe
            dimensions = f"{args.height}x{args.width}_{args.num_steps}"
            conditioning_generator = ConditioningGenerator(pipe)
            image_generator = ImageGenerator(
                pipe,
                args.width,
                args.height,
                args.num_images,
                args.num_steps,
                args.prompt,
                args.nprompt,
                conditioning_generator,
                args.show,
            )
            image_saver = PNGImageSaver(
                args.prompt, args.lora_model, dimensions, args.output_dir, args.base_path
            )
            app = Application(image_saver, image_generator)
            app.run()


        if __name__ == "__main__":
            un, pn, hn = (
                coloredlogs.find_username(),
                coloredlogs.find_program_name(),
                coloredlogs.find_hostname(),
            )
            logger = Logger(pn).logger
            logger.info(
                colored(
                    f"<{'*' * 50 }\tWelcome, {un}\t\tRunning: ./{pn} \t*** HostName: {hn}\t{'*' * 50 }",
                    "yellow",
                )
            )
            try:
                main()
            except Exception as error:
                logger.error(colored(f"ERROR :{error}", "red"))
            finally:
                logger.critical(colored(f"{'*' *20 }\tAdios, {un}!\t{'*' *20 }", "yellow"))
        