"""Main Streamlit entry point for the SuperMart Intelligence Suite."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.sidebar import render_sidebar
from pages.deep_learning import render_deep_learning_page
from pages.home import render_home


def _load_css() -> None:
	css_path = Path(__file__).resolve().parent / "assests" / "style.css"
	st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_placeholder(title: str, description: str) -> None:
	st.markdown(
		f"""
		<div class="welcome">
			<h1>{title}</h1>
			<p>{description}</p>
		</div>
		<div class="card">
			<p class="card-title">Module unavailable</p>
			<p class="card-subtitle">This workspace currently exposes the Home and Deep Learning dashboards.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)


def main() -> None:
	st.set_page_config(page_title="SuperMart Intelligence Suite", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")
	# _load_css()

	requested_page = st.query_params.get("page", "home")
	if requested_page in {"home", "deep_learning", "churn", "segmentation", "basket", "ads", "sentiment"}:
		st.session_state["current_page"] = requested_page
	st.session_state.setdefault("current_page", "home")
	render_sidebar(st.session_state["current_page"])

	page = st.session_state.get("current_page", "home")
	if page == "home":
		render_home()
	elif page == "deep_learning":
		render_deep_learning_page()
	elif page == "churn":
		_render_placeholder("Customer Churn", "Churn-specific UI is not implemented in this workspace yet.")
	elif page == "segmentation":
		_render_placeholder("Customer Segmentation", "Segmentation-specific UI is not implemented in this workspace yet.")
	elif page == "basket":
		_render_placeholder("Market Basket", "Market basket-specific UI is not implemented in this workspace yet.")
	elif page == "ads":
		_render_placeholder("Ad Optimization", "Ad optimization-specific UI is not implemented in this workspace yet.")
	elif page == "sentiment":
		_render_placeholder("Sentiment Analysis", "Sentiment-analysis-specific UI is not implemented in this workspace yet.")
	else:
		_render_placeholder("SuperMart Intelligence Suite", "Select a module from the sidebar to continue.")


if __name__ == "__main__":
	main()
