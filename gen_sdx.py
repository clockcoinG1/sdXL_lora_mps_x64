#!/Users/clockcoin/anaconda3/bin/python3.11
from colorama import Fore, Style
from abc import ABC, abstractmethod
import math, os, argparse, datetime, time, logging, json, logging, coloredlogs
from compel import Compel, ReturnedEmbeddingsType
from diffusers import  DiffusionPipeline , loaders, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput 
from diffusers.utils.pil_utils import make_image_grid
import torch
import accelerate
import python_coreml_stable_diffusion.pipeline as coreml_pipeline
def init_logger(name):
    # coloredlogs.install("DEBUG")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    coloredlogs.install(level='DEBUG', logger=logger)
    return logger


class ImageSaver(ABC):
    @abstractmethod
    def save(self, images):
        pass

class PNGImageSaver(ImageSaver):
    def __init__(self,  prompt, lora_model, dimensions , output_dir, base_path):
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
            timestamp = now.strftime('%Y%m%d%H%M%S%f')
            unique_name = f"sdxl-{self.lora_model.split('/')[-1]}{self.dimensions}_{timestamp}_{str(self.image_count)}.png"  
            unique_name = os.path.join(self.output_dir, unique_name)  
            image.save(unique_name)
            logger.warning('*' * 25)
            logger.debug(f"Image #{str(self.image_count)} saved with as: {unique_name}\t!")
            logger.debug(f"Prompt: {self.prompt[0:75]}\tSettings: {self.dimensions}\t")
            logger.debug(f"Lora Model: {self.lora_model}")
            logger.warning('*' * 25)
            self.image_count += 1 
class PipeClass:
    def __init__(self, lora_model, base_path):
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            # custom_pipeline="~/ml-stable-diffusion/python_coreml_stable_diffusion/",
            variant="fp16",
            safety_checker=None,
            use_safetensors=True,
        ).to("mps")
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        accelerate.init_on_device(device='mps', include_buffers=True)
        self.pipe.load_lora_weights(base_path, weight_name=lora_model)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            solve_order=2,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

class ConditioningGenerator:
    def __init__(self, pipe):
        from compel import DownweightMode
        self.compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True], device="mps",downweight_mode=DownweightMode.MASK)

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
        self.guidance_scale=8.0
        self.conditioning_generator = conditioning_generator

    def generate_images(self, prompt, nprompt):
        conditioning, pooled, nconditioning, npooled = self.conditioning_generator.generate_conditioning(prompt, nprompt)
        for _ in range(self.num_images):
            image = self.pipe(
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
            for img in image:
                logger.info(f"{Fore.GREEN}✅ Image generated with {self.num_steps} steps\t📝 prompt: {prompt[0:75]}")
                logger.info(f"nprompt: 📝 {nprompt[0:75]}")
                logger.info(f"🖼️ filename: {img}\t")
                yield img
            logger.info(Fore.YELLOW + f"🔍 Running iteration #{_}/{self.num_images }/ {self.num_images  * self.num_images} images generated with {self.num_steps}" + Style.RESET_ALL)

class Application:
    def __init__(self, image_saver, image_generator):
        self.image_saver = image_saver
        self.image_generator = image_generator


    def run(self, prompt, nprompt):
        
        num_images_saved = 0
        for image in self.image_generator.generate_images(prompt, nprompt):
            self.image_saver.save([image])
            num_images_saved += 1
        print(Fore.GREEN + f"{num_images_saved} images saved successfully" + Style.RESET_ALL)

        
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=f"{os.getcwd()}")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--lora_model", type=str, default=f"{os.getcwd()}/Desktop/genSD/.tensors/nsfw-sdxl.safetensors")
    parser.add_argument("--nprompt", type=str, default="(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), ugly, deformed, (extra arms:1.2)")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=f"{os.getcwd()}/media/")
    parser.add_argument("--prompt", type=str, default="....")
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    if args.num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if args.width < 1:
        raise ValueError("width must be at least 1")
    if args.height < 1:
        raise ValueError("height must be at least 1")
    if args.num_images < 1:
        raise ValueError("num_images must be at least 1")
    return args

def run_application(args):
    lora_model= args.lora_model
    pipe = PipeClass(lora_model, args.base_path).pipe
    variables_dict = {
        "width": args.width,
        "height": args.height,
        "num_steps": args.num_steps,
        "num_images": args.num_images,
        "prompt": args.prompt,
        "nprompt": args.nprompt,
        "conditioning_generator" : ConditioningGenerator(pipe),
    }
    dimensions=f'{args.height}x{args.width}_{args.num_steps}'
    image_generator = ImageGenerator(pipe, **variables_dict)
    image_saver = PNGImageSaver(args.prompt,args.lora_model, dimensions, args.output_dir, args.base_path)
    app = Application(image_saver, image_generator)
    app.run(args.prompt, args.nprompt)

def main():
    args = parse_arguments()
    run_application(args)

if __name__ == "__main__":
    logger = init_logger(__name__)
    try:
        un = coloredlogs.find_username()
        pn = coloredlogs.find_program_name()
        hn = coloredlogs.find_hostname()
        logger.info(f"< {'*' * 50 }starting {'*' * 50 }>")
        logger.info(f"Welcome, {un}\t\tRunning: ./{pn} \t***")
        logger.info(f"HostName: {hn}")
        main()
    except Exception as error: 
        logger.error(f"ERROR :({error}")
    finally:
        logger.critical(f"Good-BYE {un}!")
