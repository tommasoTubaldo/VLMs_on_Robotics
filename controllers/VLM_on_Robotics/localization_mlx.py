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


'''
print("\n\nModel:", model,"  Temperature:", temperature,"   Max Tokens:", max_tokens)
print(f"Total Inference Time: {comp_time:.2f} [s]")
print("\nPREDICTION:",textwrap.fill(prediction, width=150))
'''
