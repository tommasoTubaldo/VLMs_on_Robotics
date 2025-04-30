import textwrap
from Vlm.vlm_gemini import GeminiAPI
from Robot_Routines.robot_routines import Robot_Routines
from PIL import Image

# Create the Robot instance
robot = Robot_Routines()

# Create the VLX instance and pull the model from MLX Community on HF
model = "gemini-2.0-flash"
temperature = 0.2   # Default: 1.0, Recommended: 0.7
max_tokens = 10000
vlm = GeminiAPI(model, temperature, max_tokens)


#####   MAIN   #####
user_prompt = input("\nProvide a user prompt: ")

# VLM MLX Inference
prediction, comp_time = vlm.generate(robot, user_prompt)

print("""\n\n-----  RESULTS  -----
    \nModel:""", model,"  Temperature:", temperature,"   Max Tokens:", max_tokens)
print(f"Total Inference Time: {comp_time:.2f} [s]")
print("\nPREDICTION:",textwrap.fill(prediction.text, width=150))

