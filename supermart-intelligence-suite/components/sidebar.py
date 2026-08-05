"""Sidebar component for SuperMart SIS.

Renders the dark sidebar: logo, navigation, support section, and
the bottom profile box. The current page is highlighted.
"""

import streamlit as st

from utils.sample_data import NAV_ITEMS, SUPPORT_ITEMS


def _nav_button(label: str, key: str, active: bool):
    """Render a single sidebar navigation button."""
    if active:
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if st.button(label, key=f"nav_{key}", use_container_width=True):
        st.session_state["current_page"] = key
        st.rerun()
    if active:
        st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(current_page: str):
    """Render the full sidebar.

    Args:
        current_page: key of the page currently active (e.g. "home").
    """
    st.session_state.setdefault("current_page", current_page)
    with st.sidebar:
        # Logo + brand
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.6rem;padding:0.25rem 0 0.75rem 0;">
                <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="36" height="36" rx="10" fill="#1976D2"/>
                    <path d="M10 13h16l-1.5 9.5a2 2 0 0 1-2 1.7H13.5a2 2 0 0 1-2-1.7L10 13z" fill="#fff"/>
                    <path d="M14 13v-1.5a4 4 0 0 1 8 0V13" stroke="#fff" stroke-width="1.6" fill="none"/>
                    <circle cx="18" cy="20" r="1.6" fill="#1976D2"/>
                </svg>
                <div>
                    <div style="color:#fff;font-weight:700;font-size:1rem;line-height:1.1;">SuperMart SIS</div>
                    <div style="color:#9CA3AF;font-size:0.75rem;">Enterprise Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Navigation
        st.markdown('<p class="sidebar-label">Navigation</p>', unsafe_allow_html=True)
        for label, key in NAV_ITEMS:
            _nav_button(label, key, key == current_page)

        # Support
        st.markdown('<p class="sidebar-label">Support</p>', unsafe_allow_html=True)
        for label, key in SUPPORT_ITEMS:
            _nav_button(label, key, key == current_page)

        st.markdown("---")

        # Profile box
        st.markdown(
            """
            <div class="profile-box">
                <div class="profile-avatar">A</div>
                <div>
                    <p class="profile-name">Admin User</p>
                    <p class="profile-tier">Enterprise Tier</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
