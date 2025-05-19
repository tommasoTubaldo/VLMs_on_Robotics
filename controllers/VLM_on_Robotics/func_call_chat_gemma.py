import asyncio
from colorama import init, Fore
from Vlm.vlm_gemma import GemmaAPI
from Robot_Routines.robot_routines import Robot_Routines
init(autoreset=True)

## Tasks:
# 1. In the room there is a ball. What sport does it belong to?
# 2. What breed is the dog?

# Init
robot = Robot_Routines()
vlm = GemmaAPI(
    model="gemma-3-27b-it",
    #model="gemma-3-12b-it",
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