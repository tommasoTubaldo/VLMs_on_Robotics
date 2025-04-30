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

        # Define the system instructions and the available tools
        self.system_instruction = """
        You are an AI Agent integrated into a mobile robot (TurtleBot3) operating in a physical environment.

        You receive user commands in natural language and must reason about the scene, environment, and robot state to assist the user effectively.
        
        Your core capabilities include:
        
        - Interpreting the surrounding scene using the robot’s camera.
        - Accessing internal robot sensors (e.g., GPS) through available tools when needed.
        - Controlling the robot by setting the absolute linear position of the left and right wheels using the set_position tool.
        - Providing intelligent, context-aware responses based on visual inputs, sensor data, or both.
        - Asking the user for clarification if a request is ambiguous, lacks necessary details (e.g., distance to move), or could have multiple interpretations.
        
        Movement Instructions:
        
        - When the user issues a command to move the robot (e.g., "move forward", "go backward", "turn left", "rotate right"), you must invoke the set_position tool.
        - Use suitable values (in meters) for delta_right and delta_left to achieve the desired motion.
          - Move forward → both deltas positive and equal (e.g., 0.2).
          - Move backward → both deltas negative and equal.
          - Turn left → right delta greater than left delta (e.g., right=0.2, left=0.05).
          - Turn right → left delta greater than right delta.
          - Rotate in place → set deltas to opposite signs (e.g., left=0.1, right=-0.1).
        - If the user does not specify a distance or rotation magnitude, default to small increments (e.g., 0.2 meters).
        - If uncertain about the user's intent (e.g., unclear which direction or how far), politely ask for clarification.
        
        Always translate user intent into concrete actions using available tools, ensuring safe and intelligent control of the robot.
        """

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

        # Configure the model with system instructions and tools
        self.tools = types.Tool(function_declarations=[
            self.set_gps_function,
            self.set_image_function,
            self.set_position_function
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


    def chat(self, robot, conversation_history):
        """
        Generates a response using the current conversation history and tools.

        :param robot: Robot instance
        :param conversation_history: List of types.Content objects
        :return: Prediction from Gemini
        """
        start_time = time.time()

        contents = list(conversation_history)  # Copy to allow mutation

        # Call the model
        response = self.client.models.generate_content(
            model=self.model,
            config=self.config,
            contents=contents
        )

        # Check if tool call is made
        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_call = part.function_call

                if tool_call.name == "get_gps_position":
                    result = robot.get_gps_position(**tool_call.args)
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} GPS coordinates obtained: {result}")

                    function_response_part = types.Part.from_function_response(
                        name=tool_call.name,
                        response={"result": result},
                    )

                    contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                    contents.append(types.Content(role="user", parts=[function_response_part]))

                elif tool_call.name == "get_image":
                    result = robot.get_image(**tool_call.args)
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Camera image obtained.")

                    result = self.convert_bgra_2_base64(result, robot.camera.getWidth(), robot.camera.getHeight())

                    function_response_part = types.Part(
                        inline_data=types.Blob(mime_type="image/jpeg", data=result)
                    )

                    contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                    contents.append(types.Content(role="user", parts=[function_response_part]))

                elif tool_call.name == "set_position":
                    robot.set_position(**tool_call.args)
                    print(f"{Fore.MAGENTA}[Tool]{Style.RESET_ALL} Wheel position set to {tool_call.args}")

                    # Append function call and result of the function execution to contents
                    contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                    contents.append(types.Content(role="user", parts=[types.Part(text=f"Linear position of the wheels set to {tool_call.args}")]))

                # Final response after tool use
                final_response = self.client.models.generate_content(
                    model=self.model,
                    config=self.config,
                    contents=contents
                )
                end_time = time.time()
                return final_response, end_time - start_time

        # If no tool was used, return initial response
        end_time = time.time()
        return response, end_time - start_time


    async def chat_loop(self, robot):
        """Handle Gemini chat interaction in a single loop.
        """
        print(Fore.CYAN + "----------   TurtleBot3 VLM Chat Interface   ----------" + Style.RESET_ALL)
        print("Type 'exit' or 'quit' to end the session.\n")

        conversation_history = []

        while True:
            user_input = await asyncio.to_thread(input, Fore.GREEN + "User: " + Style.RESET_ALL)
            user_input = user_input.strip()

            if user_input.lower() in ["exit", "quit"]:
                print(Fore.CYAN + "\nSession ended.\n")
                for task in asyncio.all_tasks():
                    task.cancel()
                break

            conversation_history.append(
                types.Content(role="user", parts=[types.Part(text=user_input)])
            )

            # Call Gemini in non-blocking way
            response, comp_time = await asyncio.to_thread(self.chat, robot, conversation_history)
            model_content = response.candidates[0].content
            conversation_history.append(model_content)

            model_reply = "".join(
                part.text for part in model_content.parts if hasattr(part, "text") and part.text
            )

            print(Fore.BLUE + "\nAssistant:" + Style.RESET_ALL)
            print(textwrap.fill(model_reply, width=100))
            print(Fore.YELLOW + f"\n[Response time: {comp_time:.2f} seconds]\n")