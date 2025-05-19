import asyncio
from colorama import init, Fore
from Vlm.vlm_gemini import GeminiAPI
from Robot_Routines.robot_routines import Robot_Routines
init(autoreset=True)

## Tasks:
# 1. In the room there is a ball. What sport does it belong to?
# 2. What breed is the dog?

# Init
robot = Robot_Routines()
vlm = GeminiAPI(
    model="gemini-2.0-flash",
    #model="gemini-2.5-flash-preview-04-17",
    temperature=0.2,
    max_tokens=10000
)

# Main: executes robot sim and vlm as parallel processes
async def main():
    await asyncio.gather(
        robot.sim_loop(),
        vlm.chat_loop(robot)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass