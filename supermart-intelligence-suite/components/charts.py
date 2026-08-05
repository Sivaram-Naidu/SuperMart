"""Plotly charts for the SuperMart SIS dashboard."""

import plotly.graph_objects as go
import streamlit as st

from utils.sample_data import (
    REVENUE_ONLINE,
    REVENUE_STORE,
    REVENUE_WEEKS,
    RETENTION_HOURS,
    RETENTION_MATRIX,
    RETENTION_WEEKDAYS,
)


def render_revenue_chart():
    """Weekly revenue performance — grouped bar chart (Online vs In-Store)."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=REVENUE_WEEKS,
            y=REVENUE_ONLINE,
            name="Online",
            marker_color="#1976D2",
            marker_line_width=0,
            width=0.38,
        )
    )
    fig.add_trace(
        go.Bar(
            x=REVENUE_WEEKS,
            y=REVENUE_STORE,
            name="In-Store",
            marker_color="#93C5FD",
            marker_line_width=0,
            width=0.38,
        )
    )

    fig.update_layout(
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#1F2937"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#6B7280")),
        yaxis=dict(
            showgrid=True,
            gridcolor="#EEF1F6",
            zeroline=False,
            tickfont=dict(size=11, color="#6B7280"),
            title=dict(text="Revenue ($K)", font=dict(size=11, color="#6B7280")),
        ),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_retention_heatmap():
    """Customer retention heatmap — blue gradient, weekday x hour."""
    fig = go.Figure(
        data=go.Heatmap(
            z=RETENTION_MATRIX,
            x=RETENTION_HOURS,
            y=RETENTION_WEEKDAYS,
            colorscale=[
                [0.0, "#EFF6FF"],
                [0.25, "#BFDBFE"],
                [0.5, "#60A5FA"],
                [0.75, "#2563EB"],
                [1.0, "#1E40AF"],
            ],
            showscale=False,
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Retention: %{z}%<extra></extra>",
            xgap=4,
            ygap=4,
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#1F2937"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#6B7280"),
            side="bottom",
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#6B7280"),
            autorange="reversed",
        ),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
