import os, time, io, asyncio, base64, textwrap
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from colorama import Fore, Style, init
init(autoreset=True)

class GeminiAPI():
    def __init__(self, model, temperature:float = 0.7, max_tokens:int = 1e5, generate_with_tools:bool = True):
        self.model = model

        # Load system behavior instructions
        prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompts'))
        with open(os.path.join(prompt_dir, "system_instruction.txt"), "r") as f:
            self.system_instruction = f.read()

        self.conversation_history = []

        # Function declarations for the model
        self.set_gps_function = {
            "name": "get_gps_position",
            "description": "Gets a vector with three elements containing the x, y and z coordinates of the robot.",
            "parameters": {
                "type": "object",
                "properties": {}
            },
        }

        self.set_image_function = {
            "name": "get_image",
            "description": "Gets an image from the robot's camera.",
            "parameters": {
                "type": "object",
                "properties": {}
            },
        }

        self.set_position_function = {
            "name": "set_position",
            "description": "Moves the robot by imposing the absolute linear position of the right and left wheels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "right_position": {
                        "type": "number",
                        "description": "Absolute linear position of the right wheel in meters.",
                    },
                    "left_position": {
                        "type": "number",
                        "description": "Absolute linear position of the left wheel in meters.",
                    },
                },
                "required": ["right_position", "left_position"],
            },
        }

        self.set_velocity_function = {
            "name": "set_velocity",
            "description": "Moves the robot by imposing the linear velocity of the right and left wheels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "right_velocity": {
                        "type": "number",
                        "description": "Number between 0 and 1 that represents the percentage of maximum velocity of the right wheel.",
                    },
                    "left_velocity": {
                        "type": "number",
                        "description": "Number between 0 and 1 that represents the percentage of maximum velocity of the left wheel.",
                    },
                },
                "required": ["right_velocity", "left_velocity"],
            },
        }

        self.set_response_completed = {
            "name": "response_completed",
            "description": "Execute this function when the robot response is completed or when asking for clarifications in order to let the user provide a new prompt or a response.",
            "parameters": {
              "type": "object",
              "properties": {}
            },
        }

        # Configure the model with system instructions and tools
        self.tools = types.Tool(function_declarations=[
            self.set_gps_function,
            self.set_image_function,
            #self.set_position_function,
            self.set_velocity_function,
            self.set_response_completed
        ])

        # Configure function calling mode
        if generate_with_tools:
            mode = "AUTO"   # AUTO: The model decides whether to generate a natural language response or suggest a function call based on the prompt and context.
                            # ANY: The model is constrained to always predict a function call and guarantee function schema adherence.
                            #      If 'allowed_function_names' is not specified in tool_config, the model can choose from any of the provided function declarations.
                            #      If 'allowed_function_names' is provided as a list, the model can only choose from the functions in that list.
                            #      Use this mode when you require a function call in response to every prompt (if applicable)
        else:
            mode = "NONE"   # NONE: The model is prohibited from making function calls

        self.tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=mode #, allowed_function_names=[]
            )
        )

        self.config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            max_output_tokens=max_tokens,
            temperature=temperature,
            tools=[self.tools],
            tool_config=self.tool_config
        )

        # Configure the client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


    def convert_bgra_2_PIL_image(self, image, img_width, img_height):
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


    def convert_bgra_2_base64(self, image: bytes, img_width: int, img_height: int, format: str = "JPEG"):
        """
        Converts a BGRA raw image buffer to a base64-encoded string.

        :param image: Raw BGRA image buffer (bytes)
        :param img_width: Image width (int)
        :param img_height: Image height (int)
        :param format: Output format ('JPEG' or 'PNG')
        :return: Base64-encoded image string
        """
        # Type checks
        if not isinstance(image, (bytes, bytearray)):
            raise TypeError(f"Expected image to be bytes or bytearray, got {type(image).__name__}")

        if not isinstance(img_width, int) or not isinstance(img_height, int):
            raise TypeError("Image width and height must be integers.")

        if format.upper() not in ["JPEG", "PNG"]:
            raise ValueError("Unsupported format. Use 'JPEG' or 'PNG'.")

        expected_size = img_width * img_height * 4  # 4 channels for BGRA
        if len(image) != expected_size:
            raise ValueError(f"Image buffer size mismatch. Expected {expected_size} bytes, got {len(image)} bytes.")

        try:
            # Convert buffer to NumPy array
            np_image = np.frombuffer(image, dtype=np.uint8).reshape((img_height, img_width, 4))

            # Convert BGRA to RGB (drop alpha channel)
            np_image_rgb = np_image[:, :, [2, 1, 0]]

            # Convert to PIL Image
            pil_image = Image.fromarray(np_image_rgb)

            # Encode image to base64 string
            buffered = io.BytesIO()
            pil_image.save(buffered, format=format.upper())
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return base64_str

        except Exception as e:
            raise RuntimeError(f"Error converting image to base64: {str(e)}")


    def generate(self, robot, prompt, image:Image.Image = None):
        """
        Generates a prediction from the given prompt and images by leveraging to a Gemini model

        :param robot: Robot instance
        :param prompt: Input prompt
        :param image: Input image with PIL.Image format
        :param img_width: Image width
        :param img_height: Image height
        :return: Prediction leveraged from the vlm model
        """
        # Check if an image is provided
        if image is None:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )
            ]
        else:
            image_b64 = self.convert_bgra_2_base64(image, robot.camera.getWidth(), robot.camera.getHeight())
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=image_b64
                            )
                        )
                    ]
                )
            ]

        start_time = time.time()

        # Send request with function declarations
        response = self.client.models.generate_content(
            model=self.model,
            config=self.config,
            contents=contents
        )

        # Iterate over each part of the response and check for function calls
        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_call = part.function_call

                if tool_call.name == "get_gps_position":
                    result = robot.get_gps_position(**tool_call.args)
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} GPS coordinates obtained: {result}")

                    # Create a function response part
                    function_response_part = types.Part.from_function_response(
                        name=tool_call.name,
                        response={"result": result},
                    )

                    # Append function call and result of the function execution to contents
                    contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                    contents.append(types.Content(role="user", parts=[function_response_part]))

                elif tool_call.name == "get_image":
                    result = robot.get_image(**tool_call.args)
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Front camera image obtained.")

                    # Convert the image to b64
                    result = self.convert_bgra_2_base64(result, robot.camera.getWidth(), robot.camera.getHeight())

                    # Create a function response part
                    function_response_part = types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=result
                        )
                    )

                    # Append function call and result of the function execution to contents
                    contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                    contents.append(types.Content(role="user", parts=[function_response_part]))

                elif tool_call.name == "set_position":
                    robot.set_position(**tool_call.args)
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Wheel position set.")

                    # Append function call and result of the function execution to contents
                    contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                    contents.append(types.Content(role="user", parts=[types.Part(text=f"Linear position of the wheels set to {tool_call.args}")]))

                # Generate the final response
                final_response = self.client.models.generate_content(
                    model=self.model,
                    config=self.config,
                    contents=contents
                )

            else:
                final_response = response

        end_time = time.time()
        comp_time = end_time - start_time

        return final_response, comp_time


    def chat(self, robot):
        """
        Generates a response using the current conversation history and tools.

        :param robot: Robot instance
        :return: Prediction from Gemini
        """
        contents = list(self.conversation_history)

        while True:
            # Binary exponential backoff parameters
            max_retries = 10
            retry_delay = 2  # seconds (initial delay)
            retries = 0

            # Handle 429 and 503 exceptions adopting a binary exponential backoff algorithm
            while True:
                try:
                    start_time = time.time()
                    response = self.client.models.generate_content(
                        model=self.model,
                        config=self.config,
                        contents=contents
                    )
                    end_time = time.time()
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if retries < max_retries:
                            print(Fore.RED + f"[429 Error] Retrying in {retry_delay} seconds..." + Style.RESET_ALL)
                            time.sleep(retry_delay)
                            retries += 1
                            retry_delay *= 2
                        else:
                            raise RuntimeError(f"Exceeded retry limit due to repeated Server errors: {e}")
                    elif "503" in str(e) or "UNAVAILABLE" in str(e):
                        if retries < max_retries:
                            print(Fore.RED + f"[503 Error] Retrying in {retry_delay} seconds..." + Style.RESET_ALL)
                            time.sleep(retry_delay)
                            retries += 1
                            retry_delay *= 2
                        else:
                            raise RuntimeError(f"Exceeded retry limit due to repeated Server errors: {e}")
                    else:
                        raise e

            # Check if tool call is made
            for part in response.candidates[0].content.parts:
                if part.text:
                    print(Fore.BLUE + "\nAssistant:" + Style.RESET_ALL)
                    print(textwrap.fill(part.text, width=100))
                    print(Fore.YELLOW + f"[Inference time: {end_time - start_time:.2f} s]" + Style.RESET_ALL)

                    contents.append(types.Content(role="user", parts=[types.Part(text=part.text)]))

                if part.function_call:
                    tool_call = part.function_call

                    if tool_call.name == "get_gps_position":
                        result = robot.get_gps_position(**tool_call.args)
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} GPS coordinates obtained: {result}")

                        function_response_part = types.Part.from_function_response(name=tool_call.name,response={"result": result},)

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[function_response_part]))

                    elif tool_call.name == "get_image":
                        result = robot.get_image(**tool_call.args)
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Camera image obtained.")

                        result = self.convert_bgra_2_base64(result, robot.camera.getWidth(), robot.camera.getHeight())

                        function_response_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=result))

                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[function_response_part]))

                    elif tool_call.name == "set_position":
                        robot.set_position(**tool_call.args)
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Wheel position set to {tool_call.args}")

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"Linear position of the wheels set to {tool_call.args}")]))

                    elif tool_call.name == "set_velocity":
                        robot.set_velocity(**tool_call.args)
                        print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Wheel velocity set to {tool_call.args}")

                        # Append function call and result of the function execution to contents
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                        contents.append(types.Content(role="user", parts=[types.Part(text=f"Linear velocity of the wheels set to {tool_call.args}")]))

                    elif tool_call.name == "response_completed":
                        self.conversation_history = contents
                        return response


    async def chat_loop(self, robot):
        """Handle Gemini chat interaction in a single loop.
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
                    self.conversation_history.append(types.Content(role="user", parts=[types.Part(text=self.system_instruction)]))

                    # Count input tokens from conversation history
                    total_tokens = self.client.models.count_tokens(
                        model=self.model, contents=self.conversation_history
                    )
                    print(Fore.YELLOW + f"[Total input tokens: {total_tokens.total_tokens}]")

                for task in asyncio.all_tasks():
                    task.cancel()
                break

            # Add to the conversation history the current user prompt
            self.conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

            # Call Gemini in non-blocking way
            await asyncio.to_thread(self.chat, robot)