# Created by Tommaso Tubaldo - 5th March 2025
#
# TURTLEBOT3 CONTROLLER for SEMANTIC DESCRIPTION OF THE ENVIRONMENT through Ollama API

import textwrap, utils, os
from Vlm.vlm_ollama import Ollama
from Robot_Routines.robot_routines import Robot_Routines
from PIL import Image

# Create the Robot instance
robot = Robot_Routines()

# Create the LLavaPhi3 instance and pull the model from Ollama
model  = 'gemma3'
llava = Ollama(model)
# Create the personalized model
system_descr = 'You are a robot equipped with a camera inside a particular environment. You are provided with some images of the environment around you.'
llava.init_llava_model(system_descr)


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
images_lab = [Image.open(os.path.join("/Users/Administrator/Documents/create/controllers/VLM_on_Robotics/lab_images",img)) for img in image_files]


# VLM Inference
prompt = "Describe each image listing the objects present in the scene"
#prompt = "Complete the description by adding the content of the image and list each object of the scene. DESCRIPTION: "
prediction, comp_time = llava.multi_image_prediction([images[0]], img_width, img_height, prompt)


print("\n\nPREDICTION:",textwrap.fill(prediction, width=150))
print(f"\nTotal Inference Time: {comp_time:.2f} [s]")
