import time, re, json, os, asyncio, textwrap
import numpy as np
from PIL import Image
from absl.logging import exception
from google import genai
from google.genai import types
from colorama import Fore, Style, init
init(autoreset=True)


class GemmaAPI():
    def __init__(self, model, temperature:float = 0.7, max_tokens:int = 1e5):
        self.model = model
        self.start_turn_user = "<start_of_turn>user\n"
        self.start_turn_model = "<start_of_turn>model\n"
        self.end_turn = "<end_of_turn>\n"

        # Path to the prompt directory, relative to gemma_api.py
        prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompts'))

        # Load prompt files
        with open(os.path.join(prompt_dir, "system_and_tools_gemma.txt"), "r") as f:
            self.system_instruction = f.read()

        # Initialize the conversation history with the system instructions
        self.system_instr_prompt = (
            self.start_turn_user
            + self.system_instruction
            + self.end_turn
        )

        self.conversation_history = ""

        # Define the config parameters for the model
        self.config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )

        # Configure the Client
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


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


    def chat(self, robot):
        input_prompt = self.conversation_history
        input_image = None

        while True:
            input_prompt+=self.start_turn_model

            # If an image is available, provide it to the model
            if input_image:
                contents = [input_image, self.system_instr_prompt + input_prompt]
            else:
                contents = self.system_instr_prompt + input_prompt

            # Generate response from the model and measure inference time
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            end_time = time.time()

            # Save the model response in memory
            input_prompt += (response.text + self.end_turn)

            # Try to extract tool call (based on format: ```tool_code```)
            tool_call_match = re.search(r"```tool_code\s*(.*?)\s*```", response.text)

            # Extract text response of the model
            if tool_call_match:
                text_response = response.text[:tool_call_match.start()].strip()
            else:
                text_response = response.text

            # If the model response contains text, print it
            if text_response:
                print(Fore.BLUE + "\nAssistant:" + Style.RESET_ALL)
                print(textwrap.fill(text_response, width=100))
                print(Fore.YELLOW + f"[Inference Time: {end_time - start_time:.4f} s]" + Style.RESET_ALL)

            # If the model response contains a function call, then execute the corresponding tool and save the call with the associated arguments
            if tool_call_match:
                # If 'response_completed' is called, terminate the current response
                if re.search(r"response_completed\(\)",response.text[tool_call_match.start():].strip()):
                    self.conversation_history = input_prompt
                    return response.text, end_time - start_time

                # Execute the called function
                tool_call = tool_call_match.group(1).strip()
                result = eval("robot." + tool_call)

                if re.search(r"get_gps_position\(\)", tool_call):
                    input_prompt += self.start_turn_user + f'```tool_output\n{str(result).strip()}\n```' + self.end_turn
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} GPS coordinates obtained: {str(result).strip()}")
                elif re.search(r"get_image\(\)", tool_call):
                    input_image = self.convert2_PIL_image(result, robot.camera.getWidth(), robot.camera.getHeight())
                    input_prompt += self.start_turn_user + '```tool_output\n[image captured]\n```' + self.end_turn
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Camera image obtained.")
                elif re.search(r"set_velocity\((.*?)\)", tool_call):
                    args = re.search(r"set_velocity\((.*?)\)", tool_call).group(1).strip()
                    input_prompt += self.start_turn_user + f'```tool_output\n[wheel velocity set to ({args})]\n```' + self.end_turn
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Wheel velocity set to ({args})")
                else:
                    raise Exception(f"Unknown tool call: {tool_call}")


    async def chat_loop(self, robot):
        """Handle Gemma chat interaction in a single loop.
        """
        print(Fore.CYAN + "\n----------   TurtleBot3 VLM Chat Interface   ----------" + Style.RESET_ALL)
        print("Type 'exit' or 'quit' to end the session.")

        while True:
            user_input = await asyncio.to_thread(input, Fore.GREEN + "\n\nUser: " + Style.RESET_ALL)
            user_input = user_input.strip()

            # Exit condition
            if user_input.lower() in ["exit", "quit"]:
                print(Fore.CYAN + "\nSession ended.")

                if self.conversation_history:
                    # Count input tokens from conversation history
                    total_tokens = self.client.models.count_tokens(
                        model=self.model, contents=self.conversation_history
                    )
                    print(Fore.YELLOW + f"[Total input tokens: {total_tokens.total_tokens}]")

                for task in asyncio.all_tasks():
                    task.cancel()
                break

            # Add to the conversation history the current user prompt
            self.conversation_history += self.start_turn_user + user_input + self.end_turn

            # Call Gemma in non-blocking way
            await asyncio.to_thread(self.chat, robot)