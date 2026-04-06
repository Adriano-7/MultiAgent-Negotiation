import sys

sys.path.append("../")
sys.path.append(".")

import openai
from dotenv import load_dotenv
load_dotenv("../../runner/.env")

import os
os.environ["OPENAI_API_KEY"] = "g"

import os
import json
from glob import glob
from utils import *
import streamlit as st
from ratbench.constants import *
from explorer.basic_elements.game_filtering import *
from games import *

# data loading
root_dir = os.path.abspath(__file__).split("/")[:-3]
base_logs = os.path.join("/", *root_dir, ".logs")

with st.expander("Log Directory", expanded=True):
    if os.path.isdir(base_logs):
        col1, col2, col3, col4 = st.columns(4)

        sections = sorted([d for d in os.listdir(base_logs) if os.path.isdir(os.path.join(base_logs, d))])
        with col1:
            selected_section = st.selectbox("Section", sections)

        section_path = os.path.join(base_logs, selected_section)
        game_types = sorted([d for d in os.listdir(section_path) if os.path.isdir(os.path.join(section_path, d))])
        with col2:
            selected_game_type = st.selectbox(
                "Game Type",
                game_types,
                format_func=lambda x: x.split("_")[0].capitalize(),
            )

        game_path = os.path.join(section_path, selected_game_type)
        variants = sorted([d for d in os.listdir(game_path) if os.path.isdir(os.path.join(game_path, d))])
        with col3:
            selected_variant = st.selectbox("Variant", variants)

        variant_path = os.path.join(game_path, selected_variant)
        model_sizes = sorted([d for d in os.listdir(variant_path) if os.path.isdir(os.path.join(variant_path, d))])
        with col4:
            selected_model_size = st.selectbox("Model Size", ["All"] + model_sizes)

        log_dir = variant_path if selected_model_size == "All" else os.path.join(variant_path, selected_model_size)
    else:
        log_dir = st.text_input("Log Directory", value=base_logs)

    st.caption(f"Loading from: `{log_dir}`")

games = load_states_from_dir(log_dir, completed_only=False)
games_summary_df = compute_game_summary(games)
games_summary_df["list_name"] = games_summary_df[["game_name", "log_path", "is_complete"]].apply(
    lambda row: f"{'✓' if row.is_complete else '✗'} {row.game_name} - {from_timestamp_str(os.path.basename(row.log_path))} - {str(os.path.basename(row.log_path))}",
    axis=1,
)


# main page

st.write("# Conversation Explorer")

if games:
    # Selection Element
    games_summary_df = game_filter(games_summary_df)

    selected_game = st.selectbox("Which Game?", list(games_summary_df["list_name"]))
    option = st.selectbox("Which Player?", (1, 2))

    game_to_load = get_log_path_from_summary(selected_game, games_summary_df)

    with open(game_to_load) as f:
        # Load the json file
        game_state = json.load(f)

    st.write("You are looking at Player:", option)
    for index, msg in enumerate(game_state["players"][option - 1]["conversation"]):
        txtmsg = msg["content"]
        sys_prompt = True if index == 0 else False

        for c in ALL_CONSTANTS:
            txtmsg = text_formatting(txtmsg, sys_prompt)

        if sys_prompt:
            with st.expander("Check System Prompt"):
                with st.chat_message(msg["role"]):
                    st.write(txtmsg)
        else:
            with st.chat_message(msg["role"]):
                st.write(txtmsg)
