# Create by Tommaso Tubaldo - 19th March 2025
#
# MLX optimization class

import time, re, json
import numpy as np
from PIL import Image
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

class VlmMlx():
    def __init__(self, model):
        self.model_path = "mlx-community/" + model
        # Load the model
        self.model, self.processor = load(self.model_path,trust_remote_code=True)
        self.config = load_config(self.model_path,trust_remote_code=True)

        # Define a system prompt in order to provide context to the VLM
        self.system_prompt = """
        You are an AI Agent integrated into a mobile robot (TurtleBot3) operating in a physical environment. 
        You receive user commands in natural language and must reason about the scene, environment, and robot state.
        """

        # Additional system prompt
        """
        Your goal is to assist the user by:
        - Interpreting the surrounding scene using the robot's camera.
        - Accessing internal robot sensors (e.g., GPS) through available tools when needed.
        - Providing intelligent, context-aware answers based on visual inputs, sensor data, or both.
        """

        # Define the available tools for the VLM
        self.tool_prompt = """
        You have access to functions. If you decide to invoke any of the function(s), you MUST put it in the format of
        {"name": function name, "parameters": dictionary of argument name and its value}

        You SHOULD NOT include any other text in the response if you call a function

        [
          {
            "name": "get_gps_position",
            "description": "Gives a vector with three elements containing the x, y and z coordinates of the robot",
            "parameters": {
              "type": "object",
              "properties": {}
            }
          }
        ]
        """

        self.spec_prompt = """
        Based on the user prompt, choose if respond directly or use one or more functions with the format of 
        {"name": function name, "parameters": dictionary of argument name and its value}."""


    def convert2_PIL_image(self, image, img_width, img_height):
        """
        Converts BGRA images to PIL Images

        :param image: Input BGRA images
        :param img_width: Image width
        :param img_height: Image height
        :return: Converted PIL images
        """
        # Convert raw buffer to NumPy array (BGRA format)
        np_image = np.frombuffer(image, dtype=np.uint8).reshape((img_height, img_width, 4))

        # Convert BGRA to RGB by rearranging channels
        np_image_rgb = np_image[:, :, [2, 1, 0]]  # Swap B and R channels (BGR → RGB)

        return Image.fromarray(np_image_rgb)  # Convert NumPy array to PIL Image


    def generate(self, prompt, images, temperature=None, max_tokens=1e5):
        """
        Generates a prediction from the given prompt and images by leveraging to a VLM model

        :param prompt: Input prompt
        :param images: Input images
        :param temperature: Probability temperature (higher temp provides more creative responses)
        :param max_tokens: Maximum number of tokens to generate
        :return: Prediction leveraged from the vlm model
        """
        print("\nGenerating predictions...")
        # Ensure images are passed correctly
        if not isinstance(images, list):
            images = [images]  # Convert single image to a list if necessary

        # Apply chat template
        formatted_prompt = apply_chat_template(
            self.processor, self.config, prompt, num_images=len(images)
        )

        # Measuring computational time
        start = time.time()

        # Generate prediction
        if temperature is None:
            prediction = generate(self.model, self.processor, formatted_prompt, images,
                                  max_tokens=max_tokens,
                                  verbose=False)
        else:
            prediction = generate(self.model, self.processor, formatted_prompt, images,
                                  temperature = temperature,
                                  max_tokens = max_tokens,
                                  verbose=False)

        end = time.time()
        comp_time = end - start

        return prediction, comp_time


    def generate_with_tools(self, robot, user_prompt, images, temperature=None, max_tokens=1e5):
        # Construct the overall prompt for function calling
        prompt = f"""{self.system_prompt}
        {self.tool_prompt}
        User: {user_prompt}
        {self.spec_prompt}
        """

        prediction, comp_time = self.generate(prompt, images, max_tokens=max_tokens)

        # Try to extract tool call (based on format: {"name": ..., "parameters": {...}})
        tool_call_match = re.search(r'{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*{[^}]*}}', prediction)

        # Checks if a tool has been called
        if tool_call_match:
            try:
                tool_json = json.loads(tool_call_match.group(0))
                tool_name = tool_json["name"]

                # Executes each tool invoked
                if tool_name == "get_gps_position":
                    gps = robot.get_gps_position()
                    gps_result = {"x": gps[0], "y": gps[1], "z": gps[2]}

                    # Run second inference with the tool output
                    new_prompt = f"""Tool `get_gps_position` was called and returned: {gps_result}

                    Now, answer the user's original question: {user_prompt}
                    """

                    prediction, comp_time = self.generate(new_prompt, images, max_tokens=max_tokens)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Tool call parsing failed: {e}")

        return prediction, comp_time


    def generate_with_toolsV2(self, robot, user_prompt, images, temperature=None, max_tokens=1e5):
        # Construct the overall prompt for function calling
        tool_prompt = """
        You have access to functions. If you decide to invoke one or more of the function(s), you MUST return them as a list of JSON objects, like:
        [
          {"name": function name, "parameters": dictionary of argument name and its value},
          ...
        ]

        You SHOULD NOT include any other text in the response if you call functions.

        [
          {
            "name": "take_image",
            "description": "Captures an image from the robot's camera and returns it",
            "parameters": {
                "type": "object",
                "properties": {}
            },
          },
          {
            "name": "get_gps_position",
            "description": "Returns a vector with three elements containing the x, y and z coordinates of the robot",
            "parameters": {
                "type": "object",
                "properties": {}
            },
          }
        ]
        """

        spec_prompt = """Based on the user prompt, choose if you want to use one or more tools from the list, and return them in the following format:
        [{{"name": ..., "parameters": {{...}}}}, ...]
        If you don't use any tools, just answer the question."""

        prompt = f"""{self.system_prompt}
        {tool_prompt}
        User: {user_prompt}
        {spec_prompt}
        """

        prediction, comp_time = self.generate(prompt, images, max_tokens=max_tokens)
        print(f"\nInitial Prediction: {prediction}")

        # Try to extract tool calls (supporting multiple tools)
        tool_calls_match = re.search(r'\[\s*{[^]]*}\s*\]', prediction)

        tool_outputs = []
        used_images = []

        if tool_calls_match:
            try:
                tools_json = json.loads(tool_calls_match.group(0))

                for tool_call in tools_json:
                    tool_name = tool_call.get("name")

                    if tool_name == "get_gps_position":
                        gps = robot.get_gps_position()
                        gps_result = {"x": gps[0], "y": gps[1], "z": gps[2]}
                        tool_outputs.append(f"`get_gps_position` returned: {gps_result}")

                    elif tool_name == "take_image":
                        raw_img = robot.camera.getImage()
                        pil_img = self.convert2_PIL_image(raw_img, robot.camera.getWidth(), robot.camera.getHeight())
                        used_images.append(pil_img)
                        tool_outputs.append("`take_image` returned: [image captured]")

            except json.JSONDecodeError:
                print("Failed to parse tool call JSON.")

            # Second-stage prompt with tool results
            new_prompt = "\n".join(tool_outputs) + f"\n\nNow, answer the original user query: {user_prompt}"

            prediction, comp_time = self.generate(new_prompt, images=used_images, max_tokens=max_tokens)

        return prediction, comp_time