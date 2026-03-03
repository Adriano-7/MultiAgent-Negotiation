

import sys
sys.path.append(".")

import argparse
import traceback
from dotenv import load_dotenv

from ratbench.agents.qwen import QwenAgent
from ratbench.game_objects.resource import Resources
from ratbench.game_objects.goal import BuyerGoal, SellerGoal
from ratbench.game_objects.valuation import Valuation
from ratbench.constants import *
from games.buy_sell_game.game import BuySellGame

load_dotenv(".env")


def run_buysell_experiment(model_id, num_runs, seller_val, buyer_val, log_dir):
    success = 0
    errors = 0

    for i in range(num_runs):
        try:
            print(f"Run {i+1}/{num_runs} | Seller={seller_val}, Buyer={buyer_val}")

            a1 = QwenAgent(agent_name=AGENT_ONE, model_id=model_id)
            a2 = QwenAgent(agent_name=AGENT_TWO, model_id=model_id)

            c = BuySellGame(
                players=[a1, a2],
                iterations=10,
                resources_support_set=Resources({"X": 0}),
                player_goals=[
                    SellerGoal(cost_of_production=Valuation({"X": seller_val})),
                    BuyerGoal(willingness_to_pay=Valuation({"X": buyer_val})),
                ],
                player_initial_resources=[
                    Resources({"X": 1}),
                    Resources({MONEY_TOKEN: 100}),
                ],
                player_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                player_social_behaviour=["", ""],
                log_dir=log_dir,
            )

            c.run()
            success += 1
            print(f"  -> Game completed successfully ({success}/{i+1})")

        except Exception as e:
            errors += 1
            print(f"  -> ERROR in run {i+1}: {e}")
            traceback.print_exc()

    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Qwen Buy-Sell Experiments")
    parser.add_argument("--num_runs", type=int, default=30,
                        help="Number of runs per configuration (paper used ~30)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for Qwen")
    parser.add_argument("--log_base", type=str, default="./.logs/buysell_qwen_section_one",
                        help="Base log directory")
    args = parser.parse_args()

    # Configuration 1: Seller cost=40, Buyer WTP=60 (positive ZOPA, deal should happen)
    # This matches the paper's "seller_at_40" group

    print("CONFIG 1: Seller valuation=40, Buyer valuation=60 (Positive ZOPA)")
    s1, e1 = run_buysell_experiment(
        model_id=args.model_id,
        num_runs=args.num_runs,
        seller_val=40,
        buyer_val=60,
        log_dir=f"{args.log_base}/seller40_buyer60",
    )

    # Configuration 2: Seller cost=60, Buyer WTP=40 (negative ZOPA, deal shouldn't happen)
    # This matches the paper's "seller_at_60" group

    print("CONFIG 2: Seller valuation=60, Buyer valuation=40 (Negative ZOPA)")
    s2, e2 = run_buysell_experiment(
        model_id=args.model_id,
        num_runs=args.num_runs,
        seller_val=60,
        buyer_val=40,
        log_dir=f"{args.log_base}/seller60_buyer40",
    )

    print(f"Config 1 (Seller=40, Buyer=60): {s1} success, {e1} errors out of {args.num_runs}")
    print(f"Config 2 (Seller=60, Buyer=40): {s2} success, {e2} errors out of {args.num_runs}")
    print(f"Total: {s1+s2} success, {e1+e2} errors out of {args.num_runs*2}")


if __name__ == "__main__":
    main()