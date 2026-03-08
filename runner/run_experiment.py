#!/usr/bin/env python3
"""
Unified experiment runner for NegotiationArena with open-weight models.

Usage:
    python runner/run_experiment.py --config configs/experiments.yaml --experiment buysell_section_one
    python runner/run_experiment.py --config configs/experiments.yaml --experiment buysell_section_one --model "Qwen/Qwen2.5-7B-Instruct"

The --model flag is optional. If omitted, runs ALL models listed in the config.
If provided, runs only the specified model (useful for SLURM array jobs).
"""

import sys
import os

sys.path.append(".")

import argparse
import traceback
import yaml
from dotenv import load_dotenv

from ratbench.agents.hf_agent import HuggingFaceAgent
from ratbench.game_objects.resource import Resources
from ratbench.game_objects.goal import (
    BuyerGoal,
    SellerGoal,
    MaximisationGoal,
    UltimatumGoal,
)
from ratbench.game_objects.valuation import Valuation
from ratbench.constants import *

from games.buy_sell_game.game import BuySellGame
from games.trading_game.game import TradingGame
from games.ultimatum.ultimatum_multi_turn.game import MultiTurnUltimatumGame

load_dotenv(".env")


# ── helpers to sanitise model names for filesystem paths ──────────────
def _safe_name(model_id: str) -> str:
    """Turn 'Qwen/Qwen2.5-7B-Instruct' into 'qwen2.5-7b-instruct'."""
    return model_id.split("/")[-1].lower()


# ── game factories ────────────────────────────────────────────────────
def run_buysell(model_id, setup, num_runs, iterations, log_base):
    seller_val = setup["seller_val"]
    buyer_val = setup["buyer_val"]
    money = setup.get("money", 100)
    tag = f"seller{seller_val}_buyer{buyer_val}"
    model_tag = _safe_name(model_id)
    log_dir = os.path.join(log_base, model_tag, tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[buysell] Run {i+1}/{num_runs} | {model_tag} | {tag}")
            a1 = HuggingFaceAgent(agent_name=AGENT_ONE, model_id=model_id)
            a2 = HuggingFaceAgent(agent_name=AGENT_TWO, model_id=model_id)

            game = BuySellGame(
                players=[a1, a2],
                iterations=iterations,
                resources_support_set=Resources({"X": 0}),
                player_goals=[
                    SellerGoal(cost_of_production=Valuation({"X": seller_val})),
                    BuyerGoal(willingness_to_pay=Valuation({"X": buyer_val})),
                ],
                player_initial_resources=[
                    Resources({"X": 1}),
                    Resources({MONEY_TOKEN: money}),
                ],
                player_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                player_social_behaviour=["", ""],
                log_dir=log_dir,
            )
            game.run()
            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()

    print(f"  ✓ {model_tag}/{tag}: {success} ok, {errors} errors")
    return success, errors


def run_trading(model_id, setup, num_runs, iterations, log_base):
    p1_res = setup["p1_resources"]
    p2_res = setup["p2_resources"]
    model_tag = _safe_name(model_id)
    log_dir = os.path.join(log_base, model_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[trading] Run {i+1}/{num_runs} | {model_tag}")
            r1 = Resources(p1_res)
            r2 = Resources(p2_res)
            a1 = HuggingFaceAgent(agent_name=AGENT_ONE, model_id=model_id)
            a2 = HuggingFaceAgent(agent_name=AGENT_TWO, model_id=model_id)

            game = TradingGame(
                players=[a1, a2],
                iterations=iterations,
                resources_support_set=Resources({k: 0 for k in p1_res}),
                player_goals=[MaximisationGoal(r1), MaximisationGoal(r2)],
                player_initial_resources=[r1, r2],
                player_social_behaviour=["", ""],
                player_roles=[
                    f"You are {AGENT_ONE}, start by making a proposal.",
                    f"You are {AGENT_TWO}, start by responding to a trade.",
                ],
                log_dir=log_dir,
            )
            game.run()
            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()

    print(f"  ✓ {model_tag}: {success} ok, {errors} errors")
    return success, errors


def run_ultimatum(model_id, setup, num_runs, iterations, log_base):
    dollars = setup.get("dollars", 100)
    model_tag = _safe_name(model_id)
    log_dir = os.path.join(log_base, model_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[ultimatum] Run {i+1}/{num_runs} | {model_tag}")
            a1 = HuggingFaceAgent(agent_name=AGENT_ONE, model_id=model_id)
            a2 = HuggingFaceAgent(agent_name=AGENT_TWO, model_id=model_id)

            game = MultiTurnUltimatumGame(
                players=[a1, a2],
                iterations=iterations,
                resources_support_set=Resources({"Dollars": 0}),
                player_goals=[UltimatumGoal(), UltimatumGoal()],
                player_initial_resources=[
                    Resources({"Dollars": dollars}),
                    Resources({"Dollars": 0}),
                ],
                player_social_behaviour=["", ""],
                player_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                log_dir=log_dir,
            )
            game.run()
            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()

    print(f"  ✓ {model_tag}: {success} ok, {errors} errors")
    return success, errors


GAME_RUNNERS = {
    "buysell": run_buysell,
    "trading": run_trading,
    "ultimatum": run_ultimatum,
}


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NegotiationArena experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments.yaml",
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment block in the YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="(Optional) Run only this model_id. Useful for SLURM array jobs.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=None,
        help="Override number of runs from config (useful for quick tests)",
    )
    parser.add_argument(
        "--log_base",
        type=str,
        default=None,
        help="Override base log directory",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        all_configs = yaml.safe_load(f)

    if args.experiment not in all_configs:
        print(f"Experiment '{args.experiment}' not found. Available: {list(all_configs.keys())}")
        sys.exit(1)

    cfg = all_configs[args.experiment]
    game_type = cfg["game"]
    num_runs = args.num_runs or cfg["num_runs"]
    iterations = cfg["iterations"]
    setups = cfg["setups"]
    models = [args.model] if args.model else cfg["models"]
    log_base = args.log_base or f".logs/{args.experiment}"

    runner = GAME_RUNNERS.get(game_type)
    if runner is None:
        print(f"Unknown game type: {game_type}. Available: {list(GAME_RUNNERS.keys())}")
        sys.exit(1)

    # Run all combos
    total_success, total_errors = 0, 0
    for model_id in models:
        for setup in setups:
            s, e = runner(model_id, setup, num_runs, iterations, log_base)
            total_success += s
            total_errors += e

    print(f"\n{'='*50}")
    print(f"DONE: {total_success} succeeded, {total_errors} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()