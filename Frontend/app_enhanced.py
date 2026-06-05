import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from api_client import APIClient
from components import (
    render_header,
    render_metrics_cards,
    render_heatmap,
    render_stability_chart,
    render_build_timeline,
    render_failure_analysis
)
from components_advanced import (
    render_ai_recommendations,
    render_high_risk_tests,
    render_test_search,
    render_test_details,
    render_historical_trends,
    render_export_options,
    render_alert_configuration,
    render_test_suite_management,
    render_comparison_view
)
from config import BACKEND_URL

# Page configuration
st.set_page_config(
    page_title="AI Test Dashboard - Enhanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(base_url=BACKEND_URL)

api_client = get_api_client()

# Initialize session state
if 'current_build_cycle' not in st.session_state:
    st.session_state.current_build_cycle = None

if 'current_drift_phases' not in st.session_state:
    st.session_state.current_drift_phases = ['Backend', 'Frontend']

if 'filter_version' not in st.session_state:
    st.session_state.filter_version = 0

if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = False

def render_sidebar_enhanced():
    """Enhanced sidebar with filters and controls"""
    with st.sidebar:
        st.header("🔍 Filter Context")
        
        # Fetch build cycles
        build_cycles = api_client.get_build_cycles()
        
        if not build_cycles:
            st.warning("No build cycles available")
            return
        
        build_options = [bc['name'] for bc in build_cycles]
        build_ids = [bc['id'] for bc in build_cycles]
        
        # Build Cycle Dropdown
        default_index = 0
        if st.session_state.current_build_cycle:
            try:
                default_index = build_ids.index(st.session_state.current_build_cycle)
            except ValueError:
                default_index = 0
        
        selected_build_name = st.selectbox(
            "Select Build Cycle:",
            options=build_options,
            index=default_index,
            key=f"build_select_{st.session_state.filter_version}"
        )
        
        selected_build_id = build_ids[build_options.index(selected_build_name)]
        
        # Drift Phase Selection
        st.subheader("Drift Phase:")
        drift_options = ["Backend", "Frontend", "Database", "API", "Mobile"]
        selected_drift = []
        
        cols = st.columns(2)
        for i, option in enumerate(drift_options):
            col_idx = i % 2
            with cols[col_idx]:
                default_value = option in st.session_state.current_drift_phases
                if st.checkbox(
                    option,
                    value=default_value,
                    key=f"drift_{option}_{st.session_state.filter_version}"
                ):
                    selected_drift.append(option)
        
        # Apply Filters Button
        st.markdown("---")
        if st.button("🚀 Apply Filters", type="primary", use_container_width=True):
            st.session_state.current_build_cycle = selected_build_id
            st.session_state.current_drift_phases = selected_drift if selected_drift else ['Backend', 'Frontend']
            st.session_state.filter_version += 1
            st.cache_data.clear()
            st.rerun()
        
        # Auto-refresh
        st.markdown("---")
        st.subheader("🔄 Auto-Refresh")
        auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.auto_refresh_enabled,
            key=f"auto_refresh_{st.session_state.filter_version}"
        )
        
        if auto_refresh != st.session_state.auto_refresh_enabled:
            st.session_state.auto_refresh_enabled = auto_refresh
            st.rerun()
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Interval (seconds)",
                10, 300, 30, 10,
                key=f"interval_{st.session_state.filter_version}"
            )
        
        # Quick Actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.button("⬇️ Download Dataset CSV", use_container_width=True):
            import requests as _req
            try:
                r = _req.get(f"{BACKEND_URL}/api/export/dataset", timeout=60)
                if r.status_code == 200:
                    st.download_button(
                        label="💾 Save CSV",
                        data=r.content,
                        file_name="semanti_q_dataset.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Dataset not ready yet — run simulation first.")
            except Exception as e:
                st.error(f"Download failed: {e}")
        
        # Info
        st.markdown("---")
        # Phase indicator for selected cycle
        if selected_build_id and selected_build_id.startswith("cycle_"):
            cid = int(selected_build_id.split("_")[1])
            PHASE_MAP = [(0,19,"Backend","#1E3A8A"),(20,39,"Frontend","#7C3AED"),
                         (40,59,"Database","#0891B2"),(60,79,"API","#10B981"),(80,99,"Mobile","#F59E0B")]
            active_phase = next((p for s,e,p,_ in PHASE_MAP if s <= cid <= e), "Unknown")
            phase_color = next((c for s,e,p,c in PHASE_MAP if p == active_phase), "#888")
            st.markdown(
                f'<div style="padding:6px 10px;border-radius:8px;background:{phase_color};'
                f'color:white;text-align:center;font-weight:bold;">📍 {active_phase} Phase<br>'
                f'<small>Cycle {cid} · Range {next((f"{s}–{e}" for s,e,p,_ in PHASE_MAP if p==active_phase),"")}</small></div>',
                unsafe_allow_html=True
            )
            st.markdown("")

        st.markdown("""
        **💡 Tips:**
        - Select filters and click "Apply Filters"
        - Enable auto-refresh for live monitoring
        - Use tabs to explore different views
        """)

@st.cache_data(ttl=30)
def fetch_all_data(build_cycle, drift_phases_str, _version):
    """Fetch all dashboard data"""
    try:
        return {
            'metrics': api_client.get_dashboard_metrics(build_cycle, drift_phases_str),
            'heatmap': api_client.get_risk_heatmap(build_cycle, drift_phases_str),
            'stability': api_client.get_test_stability(build_cycle, drift_phases_str),
            'builds': api_client.get_recent_builds(10),
            'failures': api_client.get_failure_analysis(build_cycle, drift_phases_str),
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #1E3A8A;">🤖 AI-Powered Test Prioritization Dashboard</h1>', unsafe_allow_html=True)
    
    # Check backend
    if not api_client.health_check():
        st.error(f"⚠️ Backend not responding at {BACKEND_URL}")
        st.info("Start backend: `python Backend/main.py`")
        st.stop()
    
    # Check simulation
    sim_status = api_client.get_simulation_status()
    if not sim_status or not sim_status.get('has_data', False):
        progress_data = api_client.get_simulation_progress()
        phase = progress_data.get('phase', 'starting')
        pct = progress_data.get('pct', 0)
        cycle = progress_data.get('current_cycle', 0)
        total = progress_data.get('total_cycles', 80)

        st.warning("⏳ Backend simulation in progress...")
        st.progress(int(pct) if pct else 5, text=f"Phase: {phase} | Cycle {cycle}/{total}")
        if st.button("🔄 Check Status"):
            st.rerun()
        st.stop()
    
    # Render sidebar
    render_sidebar_enhanced()
    
    # Get current filters
    build_cycle = st.session_state.current_build_cycle
    drift_phases = st.session_state.current_drift_phases
    drift_phases_str = ','.join(drift_phases)
    
    # Display current filters
    st.info(f"📊 **Build:** {build_cycle or 'All'} | **Phases:** {drift_phases_str}")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Dashboard",
        "🤖 AI Insights",
        "🔍 Test Explorer",
        "📈 Analytics",
        "⚙️ Management",
        "📊 Baselines",
        "🔍 Explainability",
        "📋 Research"
    ])
    
    # Fetch data
    with st.spinner("Loading data..."):
        data = fetch_all_data(build_cycle, drift_phases_str, st.session_state.filter_version)
    
    if not data:
        st.error("Failed to load data")
        st.stop()
    
    st.caption(f"Last updated: {data['timestamp'].strftime('%H:%M:%S')}")
    
    # TAB 1: Dashboard
    with tab1:
        st.markdown("### 📊 Key Metrics")

        # Fetch prioritized test data for interactive cards
        with st.spinner("Loading test lists..."):
            prioritized_data = api_client.get_prioritized_tests(build_cycle, drift_phases_str)

        render_metrics_cards(data['metrics'], prioritized_data)
        
        st.markdown("---")

        # 5-Phase Drift Timeline
        st.markdown("### 🔄 5-Phase Drift Timeline")
        PHASES = [
            ("Backend",  0,  19, "#1E3A8A"),
            ("Frontend", 20, 39, "#7C3AED"),
            ("Database", 40, 59, "#0891B2"),
            ("API",      60, 79, "#10B981"),
            ("Mobile",   80, 99, "#F59E0B"),
        ]
        fig_drift = go.Figure()
        for phase, start, end, color in PHASES:
            fig_drift.add_trace(go.Bar(
                x=[end - start + 1],
                y=["Drift Phase"],
                base=[start],
                orientation="h",
                name=phase,
                marker_color=color,
                text=f"<b>{phase}</b><br>Cycles {start}–{end}",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"<b>{phase}</b><br>Cycles {start}–{end}<extra></extra>"
            ))

        # Highlight current cycle if simulation data available
        current_cycle = None
        sim_progress = api_client.get_simulation_progress()
        if sim_progress:
            current_cycle = sim_progress.get("current_cycle")

        if current_cycle is not None:
            fig_drift.add_vline(
                x=current_cycle,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Cycle {current_cycle}",
                annotation_position="top"
            )

        fig_drift.update_layout(
            barmode="stack",
            height=120,
            margin=dict(l=0, r=0, t=10, b=30),
            xaxis=dict(title="Build Cycle", range=[0, 100], tickvals=list(range(0, 101, 10))),
            yaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_drift, use_container_width=True)

        # Active phase badge
        if build_cycle and build_cycle.startswith("cycle_"):
            cid = int(build_cycle.split("_")[1])
            active = next((p for p, s, e, _ in PHASES if s <= cid <= e), None)
            if active:
                phase_color = next(c for p, s, e, c in PHASES if p == active)
                st.markdown(
                    f'<div style="display:inline-block;padding:4px 14px;border-radius:12px;'
                    f'background:{phase_color};color:white;font-weight:bold;font-size:14px;">'
                    f'Active Phase: {active} (Cycle {cid})</div>',
                    unsafe_allow_html=True
                )
                st.markdown("")

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔥 Risk Heatmap")
            render_heatmap(data['heatmap'])
        with col2:
            st.markdown("### 📈 Test Stability")
            render_stability_chart(data['stability'])
        
        st.markdown("---")
        st.markdown("### ⚠️ Failure Analysis")
        render_failure_analysis(data['failures'])

        st.markdown("---")
        st.markdown("### 🔁 Chronic Failures — Tests Failing Repeatedly")
        st.caption("Tests that have failed across 3+ build cycles with AI-derived root cause")

        min_cycles = st.slider("Minimum failure cycles", 2, 10, 3, key="chronic_min")
        chronic = api_client.get_chronic_failures(min_cycles=min_cycles)

        if not chronic:
            st.success("✅ No chronic failures detected with current threshold")
        else:
            # Summary metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Chronic Tests", len(chronic))
            c2.metric("Max Fail Cycles", max(t['fail_cycles'] for t in chronic))
            c3.metric("Worst Phase", chronic[0]['worst_phase'] if chronic else "—")

            # Bar chart — fail cycles per test
            top15 = chronic[:15]
            fig_ch = go.Figure(go.Bar(
                x=[t['fail_cycles'] for t in top15],
                y=[t['test_id'] for t in top15],
                orientation='h',
                marker=dict(
                    color=[t['fail_rate_pct'] for t in top15],
                    colorscale=[[0,"#F59E0B"],[0.5,"#EF4444"],[1,"#7F1D1D"]],
                    showscale=True,
                    colorbar=dict(title="Fail Rate %", thickness=10)
                ),
                text=[f"{t['fail_rate_pct']}%" for t in top15],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Failed in %{x} cycles<br>Fail Rate: %{text}<extra></extra>"
            ))
            fig_ch.update_layout(
                height=max(250, len(top15)*22),
                margin=dict(l=0, r=80, t=10, b=10),
                paper_bgcolor="#0f0f18", plot_bgcolor="#0f0f18",
                xaxis=dict(title="Cycles Failed", tickfont=dict(color="#888"), showgrid=True, gridcolor="#222"),
                yaxis=dict(tickfont=dict(color="#ccc"), showgrid=False),
                font=dict(color="#ccc")
            )
            st.plotly_chart(fig_ch, use_container_width=True, config={'displayModeBar': False})

            # Detail table with root cause
            st.markdown("##### Root Cause Analysis")
            df_ch = pd.DataFrame(chronic)[['test_id','fail_cycles','fail_rate_pct','worst_phase','failed_in_cycles','root_cause']]
            df_ch.columns = ['Test ID', 'Fail Cycles', 'Fail Rate %', 'Worst Phase', 'Recent Failures', 'Root Cause']
            st.dataframe(df_ch, use_container_width=True, hide_index=True)
    
    # TAB 2: AI Insights
    with tab2:
        st.markdown("### 🤖 AI-Powered Insights")
        render_ai_recommendations(api_client, 20)
        
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            risk_threshold = st.slider("Risk Threshold", 0.5, 1.0, 0.8, 0.05)
        render_high_risk_tests(api_client, risk_threshold)
    
    # TAB 3: Test Explorer
    with tab3:
        st.markdown("### 🔍 Test Explorer")
        render_test_search(api_client)
        
        st.markdown("---")
        st.subheader("📋 Test Details")
        test_id = st.text_input("Enter Test ID", placeholder="e.g., REG_0001")
        if test_id:
            render_test_details(api_client, test_id)
    
    # TAB 4: Analytics
    with tab4:
        st.markdown("### 📈 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            trend_start = st.date_input("Start", datetime.now() - timedelta(days=30))
        with col2:
            trend_end = st.date_input("End", datetime.now())
        
        if st.button("📊 Load Trends", type="primary"):
            render_historical_trends(api_client, trend_start, trend_end)
        
        st.markdown("---")
        
        build_cycles = api_client.get_build_cycles()
        if build_cycles and len(build_cycles) >= 2:
            render_comparison_view(api_client, [bc['id'] for bc in build_cycles])
        
        st.markdown("---")
        render_export_options(api_client, {'build_cycle': build_cycle, 'drift_phase': drift_phases_str})
    
    # TAB 5: Management
    with tab5:
        st.markdown("### ⚙️ Configuration & Management")
        render_alert_configuration(api_client)

        st.markdown("---")
        st.subheader("🧠 Model Status")

        ckpt        = api_client.get_checkpoint_status()
        loss_data   = api_client.get_training_loss()
        drift_events= api_client.get_drift_history()
        losses      = loss_data.get("loss_history", [])

        # ── Top status cards ─────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🗂️ Checkpoint", "✅ Saved" if ckpt.get("exists") else "❌ Missing")
        c2.metric("💾 Size", f"{ckpt.get('size_bytes', 0) // 1024} KB")
        c3.metric("🌊 Drift Events", len(drift_events))
        c4.metric("📉 Final Loss", f"{losses[-1]:.4f}" if losses else "—")

        st.caption(f"Last saved: {ckpt.get('saved_at', 'N/A')}")
        st.markdown("---")

        col_loss, col_drift = st.columns(2)

        # ── Training loss chart ───────────────────────────────────────────────
        with col_loss:
            st.markdown("##### 📉 Training Loss per Epoch")
            if losses:
                epochs = list(range(1, len(losses) + 1))
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=losses,
                    mode="lines+markers",
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.1)",
                    line=dict(color="#3B82F6", width=2.5, shape="spline"),
                    marker=dict(size=6, color="#3B82F6",
                                line=dict(color="#fff", width=1)),
                    hovertemplate="Epoch %{x}<br>Loss: <b>%{y:.4f}</b><extra></extra>"
                ))
                # Min loss marker
                min_idx = int(np.argmin(losses))
                fig_loss.add_trace(go.Scatter(
                    x=[epochs[min_idx]], y=[losses[min_idx]],
                    mode="markers+text",
                    marker=dict(size=12, color="#10B981", symbol="star",
                                line=dict(color="#fff", width=1)),
                    text=[f"Best: {losses[min_idx]:.4f}"],
                    textposition="top center",
                    textfont=dict(color="#10B981", size=10),
                    showlegend=False, hoverinfo="skip"
                ))
                fig_loss.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="#0f0f18",
                    plot_bgcolor="#0f0f18",
                    xaxis=dict(title="Epoch", tickfont=dict(color="#888"),
                               showgrid=False, titlefont=dict(color="#aaa")),
                    yaxis=dict(title="Loss", tickfont=dict(color="#888"),
                               showgrid=True, gridcolor="#1e1e2e",
                               titlefont=dict(color="#aaa")),
                    font=dict(color="#ccc"),
                    showlegend=False
                )
                st.plotly_chart(fig_loss, use_container_width=True,
                                config={'displayModeBar': False})
            else:
                st.info("No training loss data yet.")

        # ── Drift events timeline ─────────────────────────────────────────────
        with col_drift:
            st.markdown("##### 🌊 Drift Events & LR Adaptation")
            if drift_events:
                df_drift = pd.DataFrame(drift_events)

                fig_drift = go.Figure()
                fig_drift.add_trace(go.Scatter(
                    x=df_drift["cycle_id"],
                    y=df_drift["new_lr"],
                    mode="lines+markers",
                    line=dict(color="#F59E0B", width=2, shape="hv"),
                    marker=dict(size=9, color="#EF4444",
                                symbol="triangle-up",
                                line=dict(color="#fff", width=1)),
                    name="LR after drift",
                    hovertemplate=(
                        "Cycle %{x}<br>"
                        "Old LR: %{customdata:.6f}<br>"
                        "New LR: %{y:.6f}<extra></extra>"
                    ),
                    customdata=df_drift["old_lr"]
                ))
                fig_drift.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="#0f0f18",
                    plot_bgcolor="#0f0f18",
                    xaxis=dict(title="Build Cycle", tickfont=dict(color="#888"),
                               showgrid=False, titlefont=dict(color="#aaa")),
                    yaxis=dict(title="Learning Rate", tickfont=dict(color="#888"),
                               showgrid=True, gridcolor="#1e1e2e",
                               titlefont=dict(color="#aaa")),
                    font=dict(color="#ccc"),
                    showlegend=False
                )
                st.plotly_chart(fig_drift, use_container_width=True,
                                config={'displayModeBar': False})

                st.dataframe(
                    df_drift[["cycle_id", "old_lr", "new_lr"]].rename(columns={
                        "cycle_id": "Cycle", "old_lr": "LR Before", "new_lr": "LR After"
                    }),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No drift events detected yet — model is stable.")

    # TAB 6: Baselines
    with tab6:
        st.markdown("### 📊 Baseline Comparison: APFD · NAPFD · TTFF")
        st.caption("Semanti-Q vs. Random, History-Based, Coverage-Based, RETECS, and TCP-Net++ over cycles 20–99")

        baselines_data = api_client.get_baselines_apfd()
        if not baselines_data or not baselines_data.get("semanti_q"):
            st.error("Baseline data not yet available — simulation may still be running.")
            if st.button("🔄 Retry", key="retry_baselines"):
                st.rerun()
        else:
            n_cycles = len(baselines_data["semanti_q"])
            cycle_labels = list(range(20, 20 + n_cycles))
            napfd_data = api_client.get_napfd_history()
            ttff_data = api_client.get_ttff_history()

            series_map = {
                "semanti_q":     ("Semanti-Q (Proposed)", "#1E3A8A", 3),
                "retecs":        ("RETECS",               "#7C3AED", 2),
                "tcpnet":        ("TCP-Net++",             "#0891B2", 2),
                "history_based": ("History-Based",         "#10B981", 1),
                "coverage_based":("Coverage-Based",        "#F59E0B", 1),
                "random":        ("Random",                "#EF4444", 1),
            }

            # APFD chart
            fig_apfd = go.Figure()
            for key, (label, color, width) in series_map.items():
                vals = baselines_data.get(key, [])
                if vals:
                    fig_apfd.add_trace(go.Scatter(
                        x=cycle_labels[:len(vals)], y=vals, mode="lines",
                        name=label, line=dict(color=color, width=width)
                    ))
            fig_apfd.update_layout(
                title="APFD per Cycle — All Methods", xaxis_title="Build Cycle",
                yaxis_title="APFD", yaxis=dict(range=[0, 1]), height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_apfd, use_container_width=True)

            col_n, col_t = st.columns(2)
            with col_n:
                if napfd_data and napfd_data.get("semanti_q"):
                    fig_napfd = go.Figure()
                    for key, (label, color, width) in series_map.items():
                        vals = napfd_data.get(key, [])
                        if vals:
                            fig_napfd.add_trace(go.Scatter(
                                x=cycle_labels[:len(vals)], y=vals, mode="lines",
                                name=label, line=dict(color=color, width=width), showlegend=False
                            ))
                    fig_napfd.update_layout(
                        title="NAPFD per Cycle", xaxis_title="Cycle",
                        yaxis_title="NAPFD", yaxis=dict(range=[0, 1]), height=300
                    )
                    st.plotly_chart(fig_napfd, use_container_width=True)

            with col_t:
                if ttff_data and ttff_data.get("semanti_q"):
                    fig_ttff = go.Figure()
                    for key, (label, color, width) in series_map.items():
                        vals = ttff_data.get(key, [])
                        if vals:
                            fig_ttff.add_trace(go.Scatter(
                                x=cycle_labels[:len(vals)], y=vals, mode="lines",
                                name=label, line=dict(color=color, width=width), showlegend=False
                            ))
                    fig_ttff.update_layout(
                        title="TTFF per Cycle (lower = better)", xaxis_title="Cycle",
                        yaxis_title="Test Position", height=300
                    )
                    st.plotly_chart(fig_ttff, use_container_width=True)

            # Summary table
            st.markdown("#### Mean Scores Across All Cycles")
            summary_rows = []
            for key, (label, _, _) in series_map.items():
                a = baselines_data.get(key, [])
                n = napfd_data.get(key, []) if napfd_data else []
                t = ttff_data.get(key, []) if ttff_data else []
                summary_rows.append({
                    "Method": label,
                    "Mean APFD": f"{np.mean(a):.3f}" if a else "—",
                    "Mean NAPFD": f"{np.mean(n):.3f}" if n else "—",
                    "Mean TTFF": f"{np.mean(t):.1f}" if t else "—",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # TAB 7: Explainability
    with tab7:
        st.markdown("### 🔍 Model Explainability")

        col_shap, col_attn = st.columns(2)

        with col_shap:
            st.subheader("🧠 SHAP Feature Importance")
            shap_data = api_client.get_shap_summary()
            if not shap_data or not shap_data.get("top_features"):
                st.error("SHAP data not available yet.")
                if st.button("🔄 Retry SHAP", key="retry_shap"):
                    st.rerun()
            else:
                features = shap_data["top_features"]
                df_shap = pd.DataFrame(features).sort_values("mean_abs_shap")
                fig_shap = px.bar(
                    df_shap,
                    x="mean_abs_shap",
                    y="name",
                    orientation="h",
                    title=f"Top Features (Cycle {shap_data.get('cycle_id', '?')})",
                    labels={"mean_abs_shap": "Mean |SHAP|", "name": "Feature"},
                    color="mean_abs_shap",
                    color_continuous_scale="Blues"
                )
                fig_shap.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_shap, use_container_width=True)

        with col_attn:
            st.subheader("🎯 Cross-Attention Gate Values")
            attn_data = api_client.get_attention_weights()
            weights = attn_data.get("weights", [])
            if not weights:
                st.error("Attention weights not available yet.")
                if st.button("🔄 Retry Attention", key="retry_attn"):
                    st.rerun()
            else:
                # Reshape into 2D for heatmap (up to 50 samples × 1)
                display_weights = weights[:50]
                fig_attn = go.Figure(data=go.Heatmap(
                    z=[display_weights],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Gate")
                ))
                fig_attn.update_layout(
                    title=f"Attention Gates — Cycle {attn_data.get('cycle_id', '?')} (first {len(display_weights)} samples)",
                    xaxis_title="Sample Index",
                    yaxis=dict(showticklabels=False),
                    height=250
                )
                st.plotly_chart(fig_attn, use_container_width=True)
                st.caption("Values in (0,1). High = stat branch dominates; Low = semantic branch dominates.")
    
    # TAB 8: Research
    with tab8:
        st.markdown("### 📋 Research Results")

        r_tab1, r_tab2, r_tab3 = st.tabs(["🔬 Ablation Study", "⚙️ Sensitivity Analysis", "📊 Dataset Stats"])

        with r_tab1:
            st.subheader("Extended Ablation Study")
            st.caption("Mean APFD across cycles 20–99 for each configuration")
            ablation = api_client.get_ablation_results()
            if not ablation:
                st.info("Ablation study running — check back after simulation completes.")
            else:
                df_abl = pd.DataFrame(ablation).sort_values("apfd")
                fig_abl = px.bar(
                    df_abl, x="apfd", y="config", orientation="h",
                    color="apfd", color_continuous_scale="Blues",
                    title="Ablation: Mean APFD per Configuration",
                    labels={"apfd": "Mean APFD", "config": "Configuration"},
                    text="apfd"
                )
                fig_abl.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_abl.update_layout(height=380, showlegend=False,
                                      xaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_abl, use_container_width=True)
                st.dataframe(df_abl[["config", "apfd"]].sort_values("apfd", ascending=False),
                             use_container_width=True, hide_index=True)

        with r_tab2:
            st.subheader("Reward Weight Sensitivity Analysis")
            st.caption("Effect of w₁ (APFD weight) and w₂ (time weight) on mean APFD")
            sensitivity = api_client.get_sensitivity_results()
            if not sensitivity:
                st.info("Sensitivity analysis running — check back after simulation completes.")
            else:
                df_sens = pd.DataFrame(sensitivity)
                df_sens["label"] = df_sens.apply(lambda r: f"w₁={r['w1']}, w₂={r['w2']}", axis=1)
                fig_sens = px.bar(
                    df_sens, x="label", y="apfd",
                    color="apfd", color_continuous_scale="Greens",
                    title="Sensitivity: Mean APFD vs Reward Weights",
                    labels={"label": "Weight Combination", "apfd": "Mean APFD"},
                    text="apfd"
                )
                fig_sens.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_sens.update_layout(height=350, showlegend=False,
                                       yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_sens, use_container_width=True)
                st.dataframe(df_sens[["label", "apfd"]], use_container_width=True, hide_index=True)

        with r_tab3:
            st.subheader("Synthetic Dataset Realism Validation")
            st.caption("Comparing benchmark statistics against industrial CI/CD ranges")
            stats = api_client.get_dataset_stats()
            if not stats:
                st.info("Dataset stats not yet available.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Unique Tests", stats.get("unique_tests", "—"))
                    st.metric("Build Cycles", stats.get("build_cycles", "—"))
                    st.metric("Mean Failure Rate",
                              f"{stats.get('mean_failure_rate_pct', 0):.1f}%",
                              delta=f"Industrial: {stats.get('industrial_failure_range', '')}")
                with col_b:
                    st.metric("Median Exec Time",
                              f"{stats.get('median_execution_time_s', 0):.2f}s",
                              delta=f"Industrial: {stats.get('industrial_exec_range', '')}")
                    st.metric("Mean Code Churn",
                              f"{stats.get('mean_code_churn_lines', 0):.0f} lines",
                              delta=f"Industrial: {stats.get('industrial_churn_range', '')}")

                phases = stats.get("drift_phase_distribution", {})
                if phases:
                    fig_phases = px.pie(
                        values=list(phases.values()),
                        names=list(phases.keys()),
                        title="Test Distribution by Drift Phase",
                        hole=0.4
                    )
                    fig_phases.update_layout(height=300)
                    st.plotly_chart(fig_phases, use_container_width=True)

    # Debug info
    with st.expander("🔧 Debug Info"):
        st.json({
            "build_cycle": build_cycle,
            "drift_phases": drift_phases,
            "filter_version": st.session_state.filter_version,
            "total_tests": data['metrics'].get('total_tests', 0)
        })

    # Auto-refresh logic
    if st.session_state.auto_refresh_enabled:
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
