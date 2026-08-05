"""Home page — the main dashboard view."""

import streamlit as st

from components.cards import render_kpi_cards
from components.charts import render_revenue_chart, render_retention_heatmap
from components.navbar import render_navbar
from components.table import render_logs_table


def render_home():
    # Top navigation
    render_navbar()

    # Welcome section
    st.markdown(
        """
        <div class="welcome" style="margin-bottom:1.25rem;">
            <h1>Welcome back, Admin!</h1>
            <p>Here is the latest intelligence summary for SuperMart global operations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards
    render_kpi_cards()

    # Revenue performance + Customer retention side by side
    col_chart, col_heat = st.columns([3, 2])
    with col_chart:
        st.markdown(
            """
            <div class="card">
                <p class="card-title">Revenue Performance</p>
                <p class="card-subtitle">Weekly online vs in-store revenue ($K)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_revenue_chart()

    with col_heat:
        st.markdown(
            """
            <div class="card">
                <p class="card-title">Customer Retention</p>
                <p class="card-subtitle">Retention score by weekday & hour</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_retention_heatmap()

    # Recent intelligence logs
    st.markdown(
        """
        <div class="card section-card">
            <div class="section-header">
                <div>
                    <p class="card-title">Recent Intelligence Logs</p>
                    <p class="card-subtitle">Latest system & model events across modules</p>
                </div>
                <span class="section-pill">Live</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_logs_table()

    # Footer
    st.markdown(
        '<div class="footer">SuperMart SIS Enterprise Intelligence Suite © 2023</div>',
        unsafe_allow_html=True,
    )
