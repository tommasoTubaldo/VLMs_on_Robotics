# TURTLEBOT3 CONTROLLER for SEMANTIC DESCRIPTION OF THE ENVIRONMENT through MLX Optimization Lib

import textwrap, utils, os
from Vlm.vlm_mlx import VlmMlx
from Robot_Routines.robot_routines import Robot_Routines
from PIL import Image

# Create the Robot instance
robot = Robot_Routines()

# Create the VLX instance and pull the model from MLX Community on HF
model = "gemma-3-4b-it-4bit"
#model = "Qwen2.5-VL-7B-Instruct-4bit"
#model = "Qwen2.5-VL-3B-Instruct-4bit"
#model = "Qwen2-VL-2B-Instruct-4bit"
vlm = VlmMlx(model)


#####   MAIN   #####
# Execute robot routine
images = robot.rot_and_take_images()

# Save acquired images on '~/images'
i = 1
img_width = robot.camera.getWidth()
img_height = robot.camera.getHeight()
for image in images:
    img_name = "img"+str(i)+".png"
    utils.save_image(image, img_width, img_height, img_name)
    i+=1

image_files = [f for f in os.listdir("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/lab_images") if f.endswith(('.png', '.jpg', '.jpeg'))]
images_lab = [Image.open(os.path.join("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/lab_images", img)) for img in image_files]
images_lab[0].save("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/lab_img.jpeg")


# VLM MLX Inference
#prompt = "Describe each image listing the objects present in the scene"
prompt = "Describe the image listing the objects present in the scene"
images_PIL = [vlm.convert2_PIL_image(image, img_width, img_height) for image in images]
#images_PIL = utils.resize_images(images_PIL)
images_PIL = utils.resize_images(images_lab)
temperature = 0.2
max_tokens = 5000
prediction, comp_time = vlm.generate(prompt, images_PIL[0], temperature=temperature, max_tokens=max_tokens)


print("\n\nModel:", model,"  Temperature:", temperature,"   Max Tokens:", max_tokens)
print(f"Total Inference Time: {comp_time:.2f} [s]")
print("\nPREDICTION:",textwrap.fill(prediction, width=150))
