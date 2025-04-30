import os
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from PIL import Image
from gemma import gm

tokenizer = gm.text.Gemma3Tokenizer()
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

sampler = gm.text.Sampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)

image_files = [f for f in os.listdir("images") if f.endswith(('.png', '.jpg', '.jpeg'))]
images = [Image.open(os.path.join("images",img)) for img in image_files]

prompt = """<start_of_turn>user
Describe the images listing each object."""
for img in images:
    prompt+="<start_of_image>"
prompt+="<end_of_turn>model"

prediction = sampler.sample(prompt, images=images, max_new_tokens=200)