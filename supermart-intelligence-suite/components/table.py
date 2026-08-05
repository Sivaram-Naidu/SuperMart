"""Styled table for the Recent Intelligence Logs section."""

import pandas as pd
import streamlit as st

from utils.sample_data import INTELLIGENCE_LOGS


def render_logs_table():
    table_df = pd.DataFrame(INTELLIGENCE_LOGS)
    table_df = table_df[["event", "module", "status", "timestamp"]]
    table_df.rename(columns={"event": "Event Name", "module": "Module", "status": "Status", "timestamp": "Timestamp"}, inplace=True)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Event Name": st.column_config.TextColumn("Event Name", width="large"),
            "Module": st.column_config.TextColumn("Module", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
        },
    )
