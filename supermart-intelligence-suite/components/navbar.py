"""Top navigation bar for SuperMart SIS.

Renders a rounded search box, notification bell, date range selector,
and an export report button — all aligned in a single white card.
"""

import streamlit as st


def render_navbar():
    st.markdown(
        """
        <div class="navbar">
            <div class="navbar-search">
                <span>&#128269;</span>
                <input type="text" placeholder="Search insights, reports, customers..." />
            </div>
            <div class="navbar-icon-btn" title="Notifications">
                &#128276;
                <span class="notif-dot"></span>
            </div>
            <div class="navbar-date">
                <span>&#128197;</span>
                <span>Last 30 days</span>
                <span style="color:#9CA3AF;">&#9662;</span>
            </div>
            <button class="navbar-export">
                <span>&#11015;</span>
                <span>Export Report</span>
            </button>
        </div>
        """,
        unsafe_allow_html=True,
    )
