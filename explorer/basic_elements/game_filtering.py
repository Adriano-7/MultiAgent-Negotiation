import streamlit as st


def game_filter(games_summary_df):
    with st.expander("Filter Games"):
        col1, col2 = st.columns(2)

        with col1:
            filter_game_name = st.selectbox(
                "Game Type",
                games_summary_df["game_name"].unique().tolist(),
                index=None,
                placeholder="All game types",
            )

            filter_player_one = st.selectbox(
                "Model Player One",
                games_summary_df["model_1"].unique().tolist(),
                index=None,
                placeholder="Any model",
            )

            filter_player_two = st.selectbox(
                "Model Player Two",
                games_summary_df["model_2"].unique().tolist(),
                index=None,
                placeholder="Any model",
            )

        with col2:
            completed_only = st.checkbox("Completed games only", value=True)

            filter_behaviour_one = st.selectbox(
                "Behavior Player One",
                games_summary_df["behaviour_1"].unique().tolist(),
                index=None,
                placeholder="Any behavior",
            )

            filter_behaviour_two = st.selectbox(
                "Behavior Player Two",
                games_summary_df["behaviour_2"].unique().tolist(),
                index=None,
                placeholder="Any behavior",
            )

        if filter_game_name:
            games_summary_df = games_summary_df[games_summary_df["game_name"] == filter_game_name]
        if filter_player_one:
            games_summary_df = games_summary_df[games_summary_df["model_1"] == filter_player_one]
        if filter_player_two:
            games_summary_df = games_summary_df[games_summary_df["model_2"] == filter_player_two]
        if filter_behaviour_one:
            games_summary_df = games_summary_df[games_summary_df["behaviour_1"] == filter_behaviour_one]
        if filter_behaviour_two:
            games_summary_df = games_summary_df[games_summary_df["behaviour_2"] == filter_behaviour_two]
        if completed_only:
            games_summary_df = games_summary_df[games_summary_df["is_complete"].astype(str) == "True"]

        return games_summary_df
