"""CSS for the image-review Streamlit view."""
from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0b0f0e;
            --panel-bg: #121816;
            --panel-bg-2: #171f1c;
            --panel-border: rgba(224, 238, 232, 0.14);
            --text-main: #eef6f2;
            --text-soft: rgba(238,246,242,0.68);
            --accent-cyan: #5cc8d7;
            --accent-red: #ff8a80;
        }
        .stApp {
            background: #0b0f0e;
            color: var(--text-main);
        }
        section[data-testid="stSidebar"] {
            background: #101614;
            border-right: 1px solid rgba(224,238,232,0.10);
        }
        .block-container {
            max-width: none;
            padding: 0.85rem 1.1rem 2.2rem;
        }
        .app-hero {
            border: 1px solid rgba(224,238,232,0.12);
            border-radius: 8px;
            background: #121816;
            padding: 12px 14px;
            margin: 0 0 12px;
        }
        .app-hero h1 {
            margin: 0;
            font-size: 24px;
            line-height: 1.1;
            letter-spacing: 0;
        }
        .app-hero p {
            margin: 6px 0 0;
            color: var(--text-soft);
            font-size: 13px;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(224,238,232,0.12);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(18,24,22,0.72);
        }
        div[data-testid="stMetric"] {
            background: #121816;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.72rem 0.86rem;
        }
        div[data-testid="stMetric"] label {
            color: rgba(238,246,242,0.58);
        }
        div[data-testid="stMetricValue"] {
            color: var(--text-main);
            font-weight: 720;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.45rem;
        }
        div[data-testid="column"] {
            min-width: 0;
        }
        div[data-testid="stButton"] > button {
            border-radius: 8px;
            border: 1px solid rgba(224,238,232,0.16);
            background: rgba(238,246,242,0.06);
            color: var(--text-main);
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(92,200,215,0.44);
            background: rgba(92,200,215,0.12);
            color: var(--text-main);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(224,238,232,0.13);
            background: #121816;
            border-radius: 6px;
            transition: border-color 150ms ease, background 150ms ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.5rem 0.5rem 0.58rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(92,200,215,0.30);
            background: #141c19;
        }
        .viewer-card {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 8px;
            background: var(--panel-bg);
            min-height: 100%;
        }
        .viewer-caption {
            font-size: 11px;
            color: var(--text-soft);
            overflow-wrap: anywhere;
            line-height: 1.22;
            margin-top: 5px;
            padding-top: 5px;
            border-top: 1px solid rgba(224,238,232,0.08);
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .viewer-caption.compact {
            font-size: 10.5px;
            line-height: 1.15;
            margin-top: 4px;
            padding-top: 0;
            border-top: 0;
            display: block;
            overflow: hidden;
            color: rgba(238,246,242,0.64);
        }
        .viewer-caption.compact .caption-primary,
        .viewer-caption.compact .caption-secondary {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .viewer-caption.compact .caption-primary {
            color: rgba(238,246,242,0.72);
        }
        .viewer-caption.compact .caption-secondary {
            margin-top: 2px;
            color: rgba(238,246,242,0.46);
        }
        .filter-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            margin: 5px 0 1px;
        }
        .filter-chip {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            max-width: 100%;
            border-radius: 999px;
            border: 1px solid rgba(92,200,215,0.18);
            background: rgba(92,200,215,0.08);
            padding: 2px 5px;
            font-size: 10px;
            line-height: 1.25;
            color: rgba(238,246,242,0.84);
        }
        .filter-key {
            color: rgba(238,246,242,0.56);
        }
        .filter-value {
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .index-badge {
            float: right;
            min-width: 26px;
            text-align: center;
            border-radius: 999px;
            padding: 2px 6px;
            margin: 0 0 4px 4px;
            font-size: 10.5px;
            font-weight: 650;
            color: var(--text-main);
            background: rgba(92,200,215,0.16);
            border: 1px solid rgba(92,200,215,0.42);
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 6px;
            font-size: 10.5px;
            font-weight: 650;
            border: 1px solid rgba(255,255,255,0.16);
            margin-right: 3px;
            max-width: calc(100% - 36px);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .fail-pill {
            color: #ffe0dc;
            background: rgba(255,138,128,0.17);
            border-color: rgba(255,138,128,0.34);
        }
        .pass-pill {
            color: rgba(238,246,242,0.82);
            background: rgba(238,246,242,0.08);
            border-color: rgba(238,246,242,0.18);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--text-main);
        }
        [data-testid="stSidebar"] label {
            color: rgba(238,246,242,0.78);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            border-color: rgba(224,238,232,0.14);
            background: rgba(238,246,242,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
