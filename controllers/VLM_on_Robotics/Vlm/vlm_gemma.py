import time, re, json, os, asyncio, textwrap
import numpy as np
from PIL import Image
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
        with open(os.path.join(prompt_dir, "system_instruction.txt"), "r") as f:
            self.system_instruction = f.read()

        self.conversation_history = (
            self.start_turn_model
            + self.system_instruction
            + self.end_turn
        )

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

            if input_image:
                contents = [input_image, input_prompt]
            else:
                contents = input_prompt

            # Generate response from the model and measure inference time
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            end_time = time.time()

            # Try to extract tool call (based on format: {"name": ..., "parameters": {...}})
            tool_call_match = re.search(r'{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*{[^}]*}}', response.text)

            if tool_call_match:
                text_response = re.sub(r"''' json", "", response.text[:tool_call_match.start()]).strip()
            else:
                text_response = response.text

            if text_response:
                # Print the model response
                print(Fore.BLUE + "\nAssistant:" + Style.RESET_ALL)
                print(textwrap.fill(text_response, width=100))
                print(Fore.YELLOW + f"[Inference Time: {end_time - start_time:.4f} s]" + Style.RESET_ALL)
                input_prompt+=(response.text + self.end_turn)

            # If the response contains a function call, then execute the corresponding tool and save the call with the associated arguments
            if tool_call_match:
                try:
                    tool_json = json.loads(tool_call_match.group(0))
                    tool_name = tool_json["name"]
                    tool_args = tool_json["parameters"]

                    if tool_name == "get_gps_position":
                        gps = robot.get_gps_position(**tool_args)
                        result = {"x": gps[0], "y": gps[1], "z": gps[2]}
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} GPS coordinates obtained: {result}")

                        input_prompt+=self.start_turn_user + f"Tool get_gps_position was called and returned: {result}" + self.end_turn

                    elif tool_name == "get_image":
                        result = robot.get_image(**tool_args)
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Camera image obtained.")
                        result = self.convert2_PIL_image(result, robot.camera.getWidth(), robot.camera.getHeight())

                        input_image = result
                        input_prompt+=self.start_turn_user + "Tool get_image was called and returned: [image captured]" + self.end_turn

                    elif tool_name == "set_position":
                        robot.set_position(**tool_args)
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Wheel position set to {tool_args}")

                        #contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        #contents.append(types.Content(role="user", parts=[types.Part(text=f"Linear position of the wheels set to {tool_call.args}")]))

                    elif tool_name == "set_velocity":
                        robot.set_velocity(**tool_args)
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Wheel velocity set to {tool_args}")

                        input_prompt+=self.start_turn_user + f"Tool set_velocity was called and wheel velocity was set to: {tool_args}" + self.end_turn

                    elif tool_name == "response_completed":
                        # Append model reply before returning
                        self.conversation_history = input_prompt
                        return response, end_time - start_time

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Tool call parsing failed: {e}")


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
            self.conversation_history+=self.start_turn_user + user_input + self.end_turn
            #self.conversation_history += f"\n\nUser: {user_input}"

            # Call Gemma in non-blocking way
            await asyncio.to_thread(self.chat, robot)