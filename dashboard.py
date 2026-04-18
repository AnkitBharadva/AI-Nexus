import streamlit as st
import cv2
import pandas as pd
from main import run_traffic_simulation

st.set_page_config(page_title="Traffic Management Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.title("🚦 4-Way Adaptive Traffic Management System")
st.markdown("Real-time visual monitoring and analytics for a multi-lane adaptive traffic intersection.")

# Define Layout Components
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Intersection Feed", anchor=False)
    video_placeholder = st.empty()

with col2:
    st.subheader("Live Analytics", anchor=False)
    status_placeholder = st.empty()
    st.markdown("### Vehicles Per Lane")
    lane_chart_placeholder = st.empty()
    st.markdown("### Vehicle Type Distribution")
    type_chart_placeholder = st.empty()

def render_dashboard():
    # Initialize the generator
    simulator = run_traffic_simulation()
    
    # Process frames dynamically
    for grid, analytics in simulator:
        # Convert BGR to RGB for Streamlit rendering
        grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        video_placeholder.image(grid_rgb, channels="RGB", use_container_width=True)
        
        # Destructure analytics
        active_lane = analytics['active_lane']
        state = analytics['state']
        remaining = analytics['remaining_time']
        
        # Define color based on intersection state for frontend
        state_color = "🟢" if state == "GREEN" else ("🟡" if state == "YELLOW" else "🔴")
        
        with status_placeholder.container():
            st.metric(
                label="Intersection Status", 
                value=f"{state_color} {active_lane} is {state}", 
                delta=f"{remaining:.1f}s Remaining in Phase", 
                delta_color="off"
            )
            total_vehicles = sum(analytics['lane_counts'])
            st.write(f"**Total Vehicles Passing:** {total_vehicles}")
            
        # Draw Lane Distribution Bar Chart
        lane_df = pd.DataFrame({
            "Lane": analytics['lane_names'],
            "Vehicle Count": analytics['lane_counts']
        })
        lane_chart_placeholder.bar_chart(lane_df.set_index("Lane"), color="#00a8ff", height=250)
        
        # Draw Vehicle Type Distribution Bar Chart
        type_df = pd.DataFrame({
            "Vehicle Type": [k.capitalize() for k in analytics['type_counts'].keys()],
            "Count": list(analytics['type_counts'].values())
        })
        type_chart_placeholder.bar_chart(type_df.set_index("Vehicle Type"), color="#e84118", height=250)

if __name__ == "__main__":
    render_dashboard()
