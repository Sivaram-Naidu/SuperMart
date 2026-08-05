"""KPI metric cards for the dashboard home page."""

import streamlit as st

from utils.sample_data import KPI_CARDS

# Inline SVG icons keyed by name in sample_data.KPI_CARDS
_ICONS = {
    "users": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "revenue": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
    "transactions": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>',
    "accuracy": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
}


def _kpi_card(card: dict):
    icon_svg = _ICONS.get(card["icon"], "")
    trend_class = "trend-up" if card["trend_up"] else "trend-down"
    arrow = "&#9650;" if card["trend_up"] else "&#9660;"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon" style="background:{card['color']}1a;color:{card['color']}">
                    {icon_svg}
                </div>
                <p class="kpi-title">{card['title']}</p>
            </div>
            <p class="kpi-value">{card['value']}</p>
            <span class="kpi-trend {trend_class}">
                <span>{arrow}</span> {card['trend']} <span style="color:#9CA3AF;font-weight:500;">vs last month</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards():
    """Render the four KPI cards in a responsive grid."""
    cols = st.columns(4)
    for col, card in zip(cols, KPI_CARDS):
        with col:
            _kpi_card(card)
