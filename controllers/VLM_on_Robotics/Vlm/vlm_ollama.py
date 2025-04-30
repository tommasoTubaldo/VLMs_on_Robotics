# Create by Tommaso Tubaldo - 5th March 2025
#
# Ollama Class - Dependence for leverage VLMs through Ollama API

import numpy as np
import ollama, base64, io, os, time
from ollama import pull, generate, Client
from PIL import Image
from tqdm import tqdm

class Ollama():
    def __init__(self, model):
        # Pull VLM model from Ollama
        current_digest, bars = '', {}
        self.model = model
        for progress in pull(model, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest


    # Initialize the Llava-Phi3 model
    def init_llava_model(self, system_descr):
        # Create the model for semantic description of the environment
        client = Client()
        response = client.create(
            model='semantic_descriptor',
            from_=self.model,
            system=system_descr,
            stream=False,
        )
        print("Status 'creating the model':",response.status)


    # Function to convert Webots image (BGRA format) to Base64
    def convert_to_base64(self, image, width, height):
        # Convert the BGRA image to RGBA by reordering the channels
        image_np = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
        image_rgba = image_np[..., [2, 1, 0, 3]]  # Reorder to RGBA (swap BGR to RGB)

        # Create a PIL image from the RGBA numpy array
        img = Image.fromarray(image_rgba, 'RGBA')

        # Convert the image to JPEG format and save it to a buffer
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        # Return the base64 encoded JPEG image
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


    # Function to generate prediction by leveraging to the Llava-Phi3 multimodal model
    def generate_prediction(self, image_b64, prompt):
        response = ollama.chat(
            model='semantic_descriptor',
            messages=[
                {"role": "user", "content": prompt, "images": [image_b64]}
            ]
        )
        return response['message']['content']


    # Generate prediction with multi-images input, feeding all the images on a single prediction
    def multi_image_prediction(self, images, width, height, prompt):
        print(f"\nGenerating predictions...")
        # Convert images in b64 format
        images_b64 = []
        for image in images:
            images_b64.append(self.convert_to_base64(image, width, height))

        start = time.time()

        response = ollama.chat(
            model=self.model,   # Here the default model is selected
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64
                }
            ],
            options = {"temperature": 0.3}  # Define the temperature of the response, i.e. the creativity
        )

        end = time.time()
        comp_time = end - start

        return response['message']['content'], comp_time


    # Generate prediction with multi-images input in a recursive manner
    def multi_image_prediction_rec(self, images, width, height, prompt, mode):
        # Convert images in b64 format
        images_b64 = []
        for image in images:
            images_b64.append(self.convert_to_base64(image, width, height))

        rec_prompt = prompt
        responses = ''

        i = 1
        print()
        start = time.time()

        for image in images_b64:
            #utils.print_with_ellipsis(f"Processing image {i}")
            print(f"Processing image {i} of {len(images_b64)}...")
            i+=1
            response = generate(
                self.model,     # Here the default model is selected
                rec_prompt,
                images=[image],
                stream=False,
                options={
                    "temperature": 0.2,
                    'top_p': 0.9,  # Consider top 90% probable tokens
                    'max_tokens': 300,  # Limit response to 500 tokens
                    'stop': [],  # Stop generation at double newlines
                    'frequency_penalty': 0.5  # Reduce word repetition
                }
            )

            if mode == 1:
                rec_prompt = prompt + response['response']
            elif mode == 2:
                rec_prompt+=response['response']

            responses+=response['response']

        summ_prompt = 'Summarize the answer: ' + responses
        summarized_resp = generate('gemma3',summ_prompt,stream=False)

        end = time.time()
        comp_time = end - start

        return summarized_resp['response'], comp_time

    # Generate prediction with multi-images input in a recursive manner
    def multi_image_prediction_batched(self, images, width, height, prompt):
        # Convert images in b64 format
        images_b64 = []
        for image in images:
            images_b64.append(self.convert_to_base64(image, width, height))

        responses = ''

        i = 1
        print()
        start = time.time()

        for image in images_b64:
            # utils.print_with_ellipsis(f"Processing image {i}")
            print(f"Processing image {i} of {len(images_b64)}...")
            i += 1
            response = generate(
                self.model,     # Here the default model is selected
                prompt,
                images=[image],
                stream=False,
                options={
                    "temperature": 0.3,
                    'top_p': 1,  # Consider top 90% probable tokens
                    'max_tokens': 300,  # Limit response to 500 tokens
                    'stop': [],  # Stop generation at double newlines
                    'frequency_penalty': 0.5  # Reduce word repetition
                }
            )

            responses += response['response']

        summ_prompt = 'Summarize the following description of an environment: ' + responses
        summarized_resp = generate('llava-phi3', summ_prompt, stream=False)

        end = time.time()
        comp_time = end - start

        return summarized_resp['response'], comp_time