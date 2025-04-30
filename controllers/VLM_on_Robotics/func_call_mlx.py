# TURTLEBOT3 CONTROLLER for SEMANTIC DESCRIPTION OF THE ENVIRONMENT through MLX Optimization Lib
import textwrap, re, json
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
user_prompt = "Where are you?"
#user_prompt = "Are you close to the coordinates x=0?"

#user_prompt = "Which are your gps coordinates?"
#user_prompt = "Take an image. What do you see?"
images = []

# VLM MLX Inference
temperature = 0.2
max_tokens = 10000
prediction, comp_time = vlm.generate_with_tools(robot, user_prompt, images)
#prediction, comp_time = vlm.generate_with_toolsV2(robot, user_prompt, images)

print("\n\nModel:", model,"  Temperature:", temperature,"   Max Tokens:", max_tokens)
print(f"Total Inference Time: {comp_time:.2f} [s]")
print("\nPREDICTION:",textwrap.fill(prediction, width=150))