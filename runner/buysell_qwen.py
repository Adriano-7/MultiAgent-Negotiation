# runner/buysell_qwen.py
import sys
sys.path.append(".")
from dotenv import load_dotenv
from ratbench.agents.qwen import QwenAgent
from ratbench.game_objects.resource import Resources
from ratbench.game_objects.goal import BuyerGoal, SellerGoal
from ratbench.game_objects.valuation import Valuation
from ratbench.constants import *
import traceback
from games.buy_sell_game.game import BuySellGame

load_dotenv(".env")

if __name__ == "__main__":
    # Define model ID (change to Qwen/Qwen3-8B when available)
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 

    for i in range(1):
        try:
            # Initialize Qwen Agents
            a1 = QwenAgent(agent_name=AGENT_ONE, model_id=MODEL_ID)
            a2 = QwenAgent(agent_name=AGENT_TWO, model_id=MODEL_ID)

            c = BuySellGame(
                players=[a1, a2],
                iterations=10,
                resources_support_set=Resources({"X": 0}),
                player_goals=[
                    SellerGoal(cost_of_production=Valuation({"X": 40})),
                    BuyerGoal(willingness_to_pay=Valuation({"X": 100})),
                ],
                player_initial_resources=[
                    Resources({"X": 1}),
                    Resources({MONEY_TOKEN: 1000}),
                ],
                player_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                player_social_behaviour=["", ""],
                log_dir="./.logs/buysell_qwen",
            )

            c.run()
        except Exception as e:
            traceback.print_exc()