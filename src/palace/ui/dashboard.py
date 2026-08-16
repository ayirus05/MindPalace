"""Streamlit dashboard for inspecting and editing Core Vault lockers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from palace.models.config import PalaceConfig
from palace.vault.manager import VaultManager


def run_dashboard(config: PalaceConfig) -> None:
    """Render the dashboard with dependencies rooted in ``config``."""
    st.set_page_config(page_title="MindPalace Core Vault", page_icon="🗝️", layout="wide")
    st.title("MindPalace Core Vault")
    st.caption("Inspect and edit exact-match memory lockers stored as JSON.")

    manager = VaultManager(config)
    system_lockers = manager.list_lockers("system")
    user_lockers = manager.list_lockers("user")

    _render_locker_inventory(system_lockers, user_lockers)
    _render_create_locker(manager)

    locker_names = manager.list_lockers("all")
    if not locker_names:
        st.info("No lockers exist yet. Create one from the sidebar.")
        return

    selected = st.sidebar.selectbox("Open locker", locker_names)
    if selected is not None:
        _render_locker_editor(config, manager, selected)


def main() -> None:
    """Load configuration and render the Streamlit application.

    Set ``PALACE_CONFIG`` to use a config file outside the current directory.
    Launch with ``streamlit run src/palace/ui/dashboard.py``.
    """
    configured_path = os.environ.get("PALACE_CONFIG")
    config_path = Path(configured_path) if configured_path else Path.cwd() / "config.yaml"
    config = (
        PalaceConfig.from_yaml(config_path)
        if config_path.is_file()
        else PalaceConfig.default_for(Path.cwd())
    )
    run_dashboard(config)


def _render_locker_inventory(system: list[str], user: list[str]) -> None:
    st.sidebar.header("Lockers")
    st.sidebar.subheader("System")
    st.sidebar.caption(", ".join(system) if system else "No system lockers")
    st.sidebar.subheader("User")
    st.sidebar.caption(", ".join(user) if user else "No user lockers")


def _render_create_locker(manager: VaultManager) -> None:
    if st.sidebar.button("Create New Locker", use_container_width=True):
        st.session_state["show_create_locker"] = True

    if not st.session_state.get("show_create_locker", False):
        return

    st.sidebar.divider()
    st.sidebar.subheader("New locker setup")
    
    locker_name = st.sidebar.text_input("Locker name", key="create_name")
    field_name = st.sidebar.text_input("Initial field", key="create_field")
    
    value_kind = st.sidebar.selectbox(
        "Value type",
        ("String", "Number", "Boolean"),
        key="create_kind",
    )
    
    # UI is now reactive! Changing the dropdown instantly swaps the widget.
    if value_kind == "Boolean":
        value = st.sidebar.checkbox("Value", key="create_val_bool")
    elif value_kind == "Number":
        # Passing an integer 0 rather than 0.0 forces integer inputs
        value = st.sidebar.number_input("Value", value=0, step=1, key="create_val_num")
    else:
        value = st.sidebar.text_input("Value", key="create_val_str")
        
    if st.sidebar.button("Save Locker", type="primary", use_container_width=True):
        if locker_name and field_name:
            try:
                manager.write_field(locker_name, field_name, value)
                st.session_state["show_create_locker"] = False
                st.sidebar.success(f"Created {locker_name}")
                st.rerun()
            except (TypeError, ValueError) as exc:
                st.sidebar.error(str(exc))
        else:
            st.sidebar.error("Name and field are required.")


def _render_locker_editor(
    config: PalaceConfig,
    manager: VaultManager,
    locker_name: str,
) -> None:
    st.subheader(locker_name)
    try:
        fields = _load_locker(config, locker_name)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return

    edited_values: dict[str, Any] = {}
    complex_values: dict[str, str] = {}
    
    # Replaced st.form with st.container to allow instant UI updates
    with st.container():
        for index, (field, value) in enumerate(fields.items()):
            widget_key = f"edit_{locker_name}_{index}_{field}"
            
            if isinstance(value, bool):
                edited_values[field] = st.checkbox(field, value=value, key=widget_key)
            elif isinstance(value, int):
                edited_values[field] = st.number_input(
                    field, value=value, step=1, key=widget_key
                )
            elif isinstance(value, float):
                edited_values[field] = st.number_input(
                    field, value=value, key=widget_key
                )
            elif isinstance(value, str):
                edited_values[field] = st.text_input(field, value=value, key=widget_key)
            else:
                complex_values[field] = st.text_area(
                    f"{field} (JSON)",
                    value=json.dumps(value, indent=2, ensure_ascii=False),
                    key=widget_key,
                )

        st.divider()
        st.markdown("##### Add a new field")
        
        col1, col2 = st.columns(2)
        with col1:
            new_field = st.text_input("Field name", key=f"new_field_{locker_name}")
        with col2:
            new_kind = st.selectbox(
                "Field type",
                ("String", "Number", "Boolean"),
                key=f"new_kind_{locker_name}",
            )
            
        if new_kind == "Boolean":
            new_value = st.checkbox("Value", key=f"new_val_bool_{locker_name}")
        elif new_kind == "Number":
            new_value = st.number_input("Value", value=0, step=1, key=f"new_val_num_{locker_name}")
        else:
            new_value = st.text_input("Value", key=f"new_val_str_{locker_name}")

        st.write("") 
        if st.button("Save Changes", type="primary"):
            try:
                for field, raw_json in complex_values.items():
                    edited_values[field] = json.loads(raw_json)
                if new_field:
                    edited_values[new_field] = new_value
                
                for field, value in edited_values.items():
                    manager.write_field(locker_name, field, value)
                    
                st.success("Changes saved successfully!")
                st.rerun()
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Changes were not saved: {exc}")


def _load_locker(config: PalaceConfig, locker_name: str) -> dict[str, Any]:
    paths = (
        config.resolve(config.vault.system_lockers_dir) / f"{locker_name}.json",
        config.resolve(config.vault.user_lockers_dir) / f"{locker_name}.json",
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise ValueError(f"Locker not found: {locker_name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in locker: {locker_name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Locker must contain a JSON object: {locker_name}")
    return value


if __name__ == "__main__":
    main()