# Created by Tommaso Tubaldo - 5th March 2025
#
# TURTLEBOT3 CONTROLLER for SEMANTIC DESCRIPTION OF THE ENVIRONMENT through MLX Optimization Lib

import textwrap, utils, os
from utils import parse_bbox, plot_image_with_bboxes
from mlx_vlm.utils import load_image
from Vlm.vlm_mlx import VlmMlx
from Robot_Routines.robot_routines import Robot_Routines
from PIL import Image

# Create the LLavaPhi3 instance and pull the model from MLX Community
#model = "gemma-3-4b-it-4bit"
model = "Qwen2.5-VL-7B-Instruct-4bit"
#model = "Qwen2.5-VL-3B-Instruct-4bit"
#model = "Qwen2-VL-2B-Instruct-8bit"
vlm = VlmMlx(model)


#####   MAIN   #####
# Load each image to be processed
image_files = [f for f in os.listdir("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/lab_images") if f.endswith(('.png', '.jpg', '.jpeg'))]
images_lab = [load_image(os.path.join("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/lab_images", img)) for img in image_files]
images_lab[0].save("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/lab_img.jpeg")
test_img = load_image("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/test_img.png")

# Reduce image resolution
images_lab_res = utils.resize_images(images_lab)
#test_img_res = utils.resize_images(test_img)
test_img_res = test_img.copy()
img_width, img_height = test_img_res.size


# VLM MLX Inference
prompt = [
    {"role": "system", "content": """
    You are a helpfull assistant to detect objects in images.
     When asked to detect elements based on a description you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax].
     When there are more than one result, answer with a list of bounding boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...].
     Response format:
     ```
     [{
        "object": "object_name",
        "bboxes": [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
     }, ...]
     ```
    """},
    {"role": "user", "content": "detect all objects in the image"}
]
temperature = 0.7
max_tokens = 1000
prediction, comp_time = vlm.generate(prompt, test_img_res, temperature=temperature, max_tokens=max_tokens)
test_img_copy = test_img_res.copy()

print("\n\nModel:",model,"  Temperature:",temperature,"   Max Tokens:",max_tokens)
print(f"Total Inference Time: {comp_time:.2f} [s]")
print("\nPREDICTION:",textwrap.fill(prediction, width=150))

objects_data = parse_bbox(prediction, model_type="qwen")
plot_image_with_bboxes(test_img_copy, bboxes=objects_data, model_type="qwen")
