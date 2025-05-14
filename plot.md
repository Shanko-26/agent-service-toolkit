Great! With those constraints—**≤200 signals** and **time series plots only**—we can design a clean, robust, and extensible architecture for both user-driven and LLM-driven plotting in Streamlit with Plotly.

---

## 1. **Plot State Model**

Define a single source of truth for the plot configuration in `st.session_state`:

```python
# Example plot config structure
st.session_state.setdefault("plot_config", {
    "channels": [],         # List of channel names to plot
    "start": 0,             # Start time (seconds)
    "end": 120,             # End time (seconds)
    "overlays": [],         # List of overlays (e.g., moving average)
    "annotations": [],      # List of annotations (e.g., threshold lines)
    "layout": {},           # Plotly layout customizations
})
```

---

## 2. **Unified Update Function**

All plot changes—whether from user UI or LLM tool call—should go through a single update function:

```python
def update_plot_config(channels=None, start=None, end=None, overlays=None, annotations=None, layout=None, action_source="user"):
    cfg = st.session_state["plot_config"]
    if channels is not None:
        cfg["channels"] = channels
    if start is not None:
        cfg["start"] = start
    if end is not None:
        cfg["end"] = end
    if overlays is not None:
        cfg["overlays"] = overlays
    if annotations is not None:
        cfg["annotations"] = annotations
    if layout is not None:
        cfg["layout"] = layout
    st.session_state["plot_config"] = cfg
    st.session_state["last_plot_action"] = action_source
```

---

## 3. **User UI Integration**

- **Signal selection:** Use a multi-select or table for users to pick channels.
- **Time range:** Use sliders or input fields.
- **Overlays/annotations:** Add checkboxes or buttons for moving averages, thresholds, etc.
- **On any change:** Call `update_plot_config(...)` with the new values.

---

## 4. **LLM Tool Call Integration**

- **Tool schema:**  
  ```json
  {
    "tool": "plot_signals",
    "args": {
      "channels": ["EngineSpeed", "VehicleSpeed"],
      "start": 0,
      "end": 60,
      "overlays": [{"type": "moving_average", "channel": "EngineSpeed", "window": 5}]
    }
  }
  ```
- **Backend handler:** When the LLM issues a tool call, parse the args and call `update_plot_config(...)` with `action_source="llm"`.

---

## 5. **Plot Rendering**

- **Always render from `st.session_state["plot_config"]`:**
  ```python
  import plotly.graph_objs as go

  def render_plot(dataframe):
      cfg = st.session_state["plot_config"]
      fig = go.Figure()
      for channel in cfg["channels"]:
          fig.add_trace(go.Scatter(
              x=dataframe["time"],
              y=dataframe[channel],
              mode="lines",
              name=channel
          ))
      # Add overlays, annotations, etc.
      # ...
      st.plotly_chart(fig, use_container_width=True)
  ```

---

## 6. **Synchronization & Feedback**

- **After LLM tool call:** Add a chat message summarizing the action (“Added EngineSpeed and VehicleSpeed to plot”).
- **After user action:** Optionally, log the action for audit or undo/redo.

---

## 7. **Session Management**

- **Streamlit session state** is per user, so each user’s plot and chat are isolated.
- **No need for external cache** unless you want cross-session sharing or persistence.

---

## 8. **Extensibility**

- **Add overlays:** Extend the `overlays` list in the config and update the plot rendering logic.
- **Add analysis tools:** LLM can call new tools (e.g., highlight anomalies), which update the plot config and trigger a re-render.

---

## 9. **Example: User and LLM Interaction**

- **User:** Selects “EngineSpeed” and “VehicleSpeed” → UI calls `update_plot_config(channels=[...])`
- **LLM:** User says “Add a moving average to EngineSpeed” → LLM tool call updates `overlays` in config
- **Both:** Plot always reflects the latest config, regardless of who made the change.

---

## 10. **Sample Code Snippet**

```python
# User UI
selected_channels = st.multiselect("Select channels", options=all_channels)
if st.button("Update Plot"):
    update_plot_config(channels=selected_channels, action_source="user")

# LLM tool call handler (pseudo-code)
def handle_llm_tool_call(tool_args):
    update_plot_config(**tool_args, action_source="llm")

# Plot rendering
render_plot(dataframe)
```

---

## 11. **Summary Table**

| Source | How to Update | State Key | How to Render | Feedback |
|--------|---------------|-----------|---------------|----------|
| User   | UI event      | plot_config | plotly_chart | Optional |
| LLM    | Tool call     | plot_config | plotly_chart | Chat msg |

---

**Ready for a concrete implementation or want to see a full code template for this pattern?**
