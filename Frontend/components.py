import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def render_header(title: str = "AI Test Dashboard"):
    """Render dashboard header"""
    st.markdown(f"<h1 style='text-align: center; color: #1E3A8A;'>{title}</h1>", 
                unsafe_allow_html=True)

def render_sidebar_filters(api_client, default_start_date=None, default_end_date=None, default_drift_phases=None) -> Optional[Dict]:
    """Render sidebar filters and return filter values"""
    from datetime import datetime, timedelta
    if default_start_date is None:
        default_start_date = datetime.now() - timedelta(days=7)
    if default_end_date is None:
        default_end_date = datetime.now()
    if default_drift_phases is None:
        default_drift_phases = ["Backend", "Frontend"]

    with st.sidebar:
        st.header("🔍 Filter Context")
        
        # Fetch build cycles from API
        build_cycles = api_client.get_build_cycles()
        build_options = [bc['name'] for bc in build_cycles] if build_cycles else ["Build 1", "Build 2", "Build 3"]
        build_ids = [bc['id'] for bc in build_cycles] if build_cycles else []
        
        # Build Cycle Dropdown
        selected_build_name = st.selectbox(
            "Select Build Cycle:",
            options=build_options,
            index=0,
            key="build_cycle_select"
        )
        
        # Get corresponding ID
        selected_build_id = None
        if build_cycles and selected_build_name:
            for bc in build_cycles:
                if bc['name'] == selected_build_name:
                    selected_build_id = bc['id']
                    break
        
        # Drift Phase Selection
        st.subheader("Drift Phase:")
        drift_options = ["Backend", "Frontend", "Database", "API", "Mobile"]
        selected_drift = []
        
        cols = st.columns(2)
        for i, option in enumerate(drift_options):
            col_idx = i % 2
            with cols[col_idx]:
                if st.checkbox(option,
                               value=option in default_drift_phases,
                               key=f"drift_{option}"):
                    selected_drift.append(option)
        
        # Date Range
        st.subheader("Date Range:")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=default_start_date)
        with col2:
            end_date = st.date_input("End", value=default_end_date)
        
        # Additional Filters
        with st.expander("Advanced Filters"):
            priority = st.select_slider(
                "Priority Level:",
                options=["all", "low", "medium", "high", "critical"],
                value="all"
            )
        
        # Apply Filters Button
        if st.button("🚀 Apply Filters", type="primary", use_container_width=True):
            return {
                'build_cycle': selected_build_id or selected_build_name,
                'drift_phase': selected_drift,
                'start_date': start_date,
                'end_date': end_date,
                'priority': priority
            }
        
        # Quick Actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("📥 Export Report", use_container_width=True):
            st.success("Report export initiated!")
        
        if st.button("🚨 Alert Settings", use_container_width=True):
            st.info("Alert settings opened")
        
        # Info Section
        st.markdown("---")
        st.markdown("""
        **💡 Tips:**
        - Select build cycle to view specific metrics
        - Filter by drift phase to focus on specific areas
        - Use advanced filters for detailed analysis
        """)
    
    return None

def render_metrics_cards(metrics: Dict, prioritized_data: Dict = None):
    """Render metric cards with expandable test lists"""
    cols = st.columns(4)

    total_tests = metrics.get('total_tests', 0)
    tests_prioritized = metrics.get('tests_prioritized', 0)
    critical_failures = metrics.get('critical_failures', 0)
    time_saved = metrics.get('time_saved_minutes', 0)

    prioritized_list = prioritized_data.get('prioritized', []) if prioritized_data else []
    critical_list = prioritized_data.get('critical_failures', []) if prioritized_data else []
    skipped_list = prioritized_data.get('skipped', []) if prioritized_data else []
    time_saved_actual = prioritized_data.get('time_saved_minutes', time_saved) if prioritized_data else time_saved

    with cols[0]:
        st.metric(label="🧪 Total Tests in Suite", value=f"{total_tests:,}")

    with cols[1]:
        pct = f"{tests_prioritized / total_tests * 100:.1f}%" if total_tests else "30.0%"
        st.metric(label="🤖 Tests Prioritized (AI)", value=f"{tests_prioritized:,}", delta=pct)
        if prioritized_list:
            with st.expander(f"View {len(prioritized_list)} prioritized tests"):
                df_p = pd.DataFrame(prioritized_list)
                cols_p = ['rank', 'test_id', 'ai_score', 'failure_probability', 'result', 'drift_phase', 'reason']
                cols_p = [c for c in cols_p if c in df_p.columns]
                df_p = df_p[cols_p].copy()
                df_p.columns = [c.replace('_', ' ').title() for c in cols_p]
                st.dataframe(df_p, use_container_width=True, hide_index=True)

    with cols[2]:
        st.metric(label="⚠️ Critical Failures Caught", value=f"{len(critical_list) if critical_list else critical_failures:,}")
        if critical_list:
            with st.expander(f"View {len(critical_list)} critical failures"):
                df_c = pd.DataFrame(critical_list)
                cols_c = ['rank', 'test_id', 'ai_score', 'failure_probability', 'drift_phase', 'reason']
                cols_c = [c for c in cols_c if c in df_c.columns]
                df_c = df_c[cols_c].copy()
                df_c.columns = [c.replace('_', ' ').title() for c in cols_c]
                st.dataframe(df_c, use_container_width=True, hide_index=True)

    with cols[3]:
        st.metric(label="⏱️ Est. Time Saved", value=f"{time_saved_actual} min", delta="High Efficiency")
        if skipped_list:
            with st.expander(f"View {len(skipped_list)} skipped tests"):
                df_s = pd.DataFrame(skipped_list)
                cols_s = ['test_id', 'ai_score', 'execution_time', 'drift_phase', 'reason']
                cols_s = [c for c in cols_s if c in df_s.columns]
                df_s = df_s[cols_s].copy()
                df_s.columns = [c.replace('_', ' ').title() for c in cols_s]
                st.dataframe(df_s, use_container_width=True, hide_index=True)

def render_heatmap(heatmap_data: Dict):
    """Render risk heatmap visualization"""
    if not heatmap_data:
        st.warning("No heatmap data available")
        return

    z_data = heatmap_data.get('data', [])
    x_labels = heatmap_data.get('x_labels', [])
    y_labels = heatmap_data.get('y_labels', [])

    if not z_data or not x_labels:
        st.warning("No heatmap data available")
        return

    z_arr = np.array(z_data)

    # Custom red-black colorscale for dark theme
    colorscale = [
        [0.0,  "rgb(15,15,25)"],
        [0.2,  "rgb(60,10,10)"],
        [0.5,  "rgb(160,30,30)"],
        [0.75, "rgb(220,60,20)"],
        [1.0,  "rgb(255,200,50)"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_arr,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            title=dict(text="Risk Score", side="right"),
            thickness=12,
            len=0.85,
            tickfont=dict(color="#aaa", size=10),
            titlefont=dict(color="#ccc", size=11)
        ),
        hovertemplate="<b>%{x}</b><br>%{y}<br>Risk: <b>%{z:.3f}</b><extra></extra>"
    ))

    # Annotate cells with values where risk > 0
    annotations = []
    for i, row in enumerate(z_arr):
        for j, val in enumerate(row):
            if val > 0.01:
                annotations.append(dict(
                    x=x_labels[j], y=y_labels[i],
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color="white" if val > 0.03 else "#888", size=9)
                ))

    fig.update_layout(
        title=dict(
            text="🔥 Churn × Failure Risk per Module",
            font=dict(size=13, color="#e0e0e0"),
            x=0
        ),
        xaxis=dict(
            title="Drift Module",
            tickfont=dict(color="#ccc", size=11),
            titlefont=dict(color="#aaa"),
            side="bottom",
            showgrid=False
        ),
        yaxis=dict(
            title="Build Cycle",
            tickfont=dict(color="#ccc", size=10),
            titlefont=dict(color="#aaa"),
            showgrid=False,
            autorange="reversed"
        ),
        annotations=annotations,
        height=420,
        margin=dict(l=60, r=20, t=45, b=50),
        plot_bgcolor="#0f0f18",
        paper_bgcolor="#0f0f18",
        font=dict(color="#ccc")
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Risk summary row below chart
    flat = z_arr.flatten()
    flat_nonzero = flat[flat > 0]
    if len(flat_nonzero):
        max_risk = flat_nonzero.max()
        mean_risk = flat_nonzero.mean()
        hot_module = x_labels[int(np.argmax(z_arr.max(axis=0)))]
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak Risk Score", f"{max_risk:.3f}")
        c2.metric("Mean Risk Score", f"{mean_risk:.3f}")
        c3.metric("Hottest Module", hot_module)

def render_stability_chart(stability_data: Dict):
    """Render test stability chart"""
    if not stability_data:
        st.warning("No stability data available")
        return

    pass_pct  = stability_data.get('pass_percentage', 96.24)
    fail_pct  = stability_data.get('fail_percentage', 3.76)
    flaky     = stability_data.get('flaky_tests', 0)
    stable    = stability_data.get('stable_tests', 0)

    # ── Gauge chart for pass rate ─────────────────────────────────────────────
    gauge_color = "#10B981" if pass_pct >= 95 else "#F59E0B" if pass_pct >= 85 else "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pass_pct,
        delta={"reference": 95, "valueformat": ".2f",
               "increasing": {"color": "#10B981"},
               "decreasing": {"color": "#EF4444"}},
        number={"suffix": "%", "font": {"size": 36, "color": "#e0e0e0"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#444",
                "tickfont": {"color": "#888", "size": 10},
                "nticks": 6
            },
            "bar": {"color": gauge_color, "thickness": 0.25},
            "bgcolor": "#1a1a2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  70], "color": "#2a0a0a"},
                {"range": [70, 85], "color": "#2a1a0a"},
                {"range": [85, 95], "color": "#0a2a1a"},
                {"range": [95, 100],"color": "#0a1a2a"},
            ],
            "threshold": {
                "line": {"color": "#F59E0B", "width": 2},
                "thickness": 0.75,
                "value": 95
            }
        },
        title={"text": "Pass Rate", "font": {"size": 13, "color": "#aaa"}}
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="#0f0f18",
        font=dict(color="#ccc")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Stacked horizontal bar: stable vs flaky vs failing ────────────────────
    total = stable + flaky + max(0, round(fail_pct / 100 * (stable + flaky), 0))
    fail_count = round(fail_pct / 100 * (stable + flaky))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Stable", x=[stable], y=["Tests"],
        orientation="h", marker_color="#10B981",
        hovertemplate="Stable: %{x}<extra></extra>"
    ))
    fig2.add_trace(go.Bar(
        name="Failing", x=[fail_count], y=["Tests"],
        orientation="h", marker_color="#EF4444",
        hovertemplate="Failing: %{x}<extra></extra>"
    ))
    if flaky > 0:
        fig2.add_trace(go.Bar(
            name="Flaky", x=[flaky], y=["Tests"],
            orientation="h", marker_color="#F59E0B",
            hovertemplate="Flaky: %{x}<extra></extra>"
        ))

    fig2.update_layout(
        barmode="stack",
        height=80,
        margin=dict(l=0, r=0, t=5, b=5),
        paper_bgcolor="#0f0f18",
        plot_bgcolor="#0f0f18",
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.1,
                    xanchor="center", x=0.5,
                    font=dict(color="#ccc", size=10)),
        showlegend=True
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # ── Bottom metrics ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Stable", f"{stable:,}")
    c2.metric("❌ Failing", f"{fail_count:,}")
    c3.metric("⚡ Flaky", f"{flaky:,}")

def render_build_timeline(builds: List[Dict]):
    """Render build timeline visualization"""
    if not builds:
        st.warning("No build data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(builds)
    
    # Create timeline chart
    fig = go.Figure()
    
    for status, color in [('passed', '#10B981'), ('failed', '#EF4444'), ('running', '#F59E0B')]:
        status_df = df[df['status'] == status]
        if not status_df.empty:
            fig.add_trace(go.Scatter(
                x=status_df['timestamp'],
                y=status_df['duration'] if 'duration' in status_df else range(len(status_df)),
                mode='markers',
                name=status.capitalize(),
                marker=dict(size=15, color=color),
                hovertemplate='<b>Build:</b> %{text}<br>' +
                            '<b>Status:</b> %{marker.color}<br>' +
                            '<b>Duration:</b> %{y}s<br>' +
                            '<b>Tests:</b> %{customdata[0]}<br>' +
                            '<extra></extra>',
                text=status_df['id'],
                customdata=status_df[['tests']].values if 'tests' in status_df else None
            ))
    
    fig.update_layout(
        title="Recent Build Timeline",
        xaxis_title="Time",
        yaxis_title="Build Duration (s)",
        yaxis=dict(rangemode="tozero"),
        height=300,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_test_distribution(test_data: Dict):
    """Render test type distribution"""
    if not test_data:
        st.warning("No test distribution data available")
        return
    
    # Create bar chart
    test_types = list(test_data.keys())
    counts = list(test_data.values())
    
    fig = px.bar(
        x=test_types,
        y=counts,
        color=test_types,
        title="Test Type Distribution",
        labels={'x': 'Test Type', 'y': 'Count'},
        text=counts
    )
    
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(height=300, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

def render_failure_analysis(failure_data: Dict):
    """Render failure analysis"""
    if not failure_data:
        st.warning("No failure analysis data available")
        return

    by_module = failure_data.get('by_module', {})
    by_type   = failure_data.get('by_type', {})
    trend     = failure_data.get('trend', [])

    # ── Row 1: module bar + type donut ───────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### 📦 Failures by Module")
        if by_module:
            modules  = list(by_module.keys())
            counts   = list(by_module.values())
            total_f  = sum(counts)
            pcts     = [c / total_f * 100 for c in counts]

            # Colour gradient: most failures = brightest red
            max_c = max(counts) if counts else 1
            bar_colors = [
                f"rgb({int(180 + 75 * c/max_c)},{int(30 - 20 * c/max_c)},{int(30 - 20 * c/max_c)})"
                for c in counts
            ]

            fig_mod = go.Figure(go.Bar(
                x=counts,
                y=modules,
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{c}  ({p:.1f}%)" for c, p in zip(counts, pcts)],
                textposition="outside",
                textfont=dict(color="#ccc", size=11),
                hovertemplate="<b>%{y}</b><br>Failures: %{x}<extra></extra>"
            ))
            fig_mod.update_layout(
                height=220,
                margin=dict(l=0, r=60, t=10, b=10),
                paper_bgcolor="#0f0f18",
                plot_bgcolor="#0f0f18",
                xaxis=dict(showgrid=True, gridcolor="#222", tickfont=dict(color="#888")),
                yaxis=dict(tickfont=dict(color="#ccc"), showgrid=False),
                font=dict(color="#ccc")
            )
            st.plotly_chart(fig_mod, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No module data")

    with col2:
        st.markdown("##### 🔬 Failures by Type")
        if by_type:
            type_colors = {
                "Assertion": "#EF4444",
                "Timeout":   "#F59E0B",
                "Network":   "#3B82F6",
                "UI":        "#8B5CF6",
            }
            labels = list(by_type.keys())
            values = list(by_type.values())
            colors = [type_colors.get(l, "#6B7280") for l in labels]

            fig_type = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0f0f18", width=2)),
                textinfo="label+percent",
                textfont=dict(size=11, color="white"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                direction="clockwise",
                sort=True
            ))
            fig_type.add_annotation(
                text=f"<b>{sum(values)}</b><br><span style='font-size:10px'>total</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#e0e0e0"),
                align="center"
            )
            fig_type.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="#0f0f18",
                showlegend=True,
                legend=dict(
                    orientation="v", x=1.02, y=0.5,
                    font=dict(color="#ccc", size=10),
                    bgcolor="rgba(0,0,0,0)"
                ),
                font=dict(color="#ccc")
            )
            st.plotly_chart(fig_type, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No type data")

    # ── Row 2: trend area chart ───────────────────────────────────────────────
    if trend:
        st.markdown("##### 📉 Failure Trend — Last 10 Cycles")
        trend_labels = failure_data.get('trend_labels', list(range(len(trend))))
        x_labels = [f"Cycle {c}" for c in trend_labels]

        fig_trend = go.Figure()

        # Gradient fill area
        fig_trend.add_trace(go.Scatter(
            x=x_labels, y=trend,
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#EF4444", width=2.5, shape="spline", smoothing=0.8),
            marker=dict(
                size=8, color="#EF4444",
                line=dict(color="#fff", width=1.5),
                symbol="circle"
            ),
            hovertemplate="<b>%{x}</b><br>Failures: <b>%{y}</b><extra></extra>"
        ))

        # Peak marker
        peak_idx = int(np.argmax(trend))
        fig_trend.add_trace(go.Scatter(
            x=[x_labels[peak_idx]], y=[trend[peak_idx]],
            mode="markers+text",
            marker=dict(size=13, color="#F59E0B", symbol="star",
                        line=dict(color="#fff", width=1)),
            text=[f"Peak: {trend[peak_idx]}"],
            textposition="top center",
            textfont=dict(color="#F59E0B", size=11),
            hoverinfo="skip",
            showlegend=False
        ))

        # Average reference line
        avg = sum(trend) / len(trend)
        fig_trend.add_hline(
            y=avg, line_dash="dot", line_color="#3B82F6", line_width=1.5,
            annotation_text=f"avg {avg:.1f}",
            annotation_font=dict(color="#3B82F6", size=11),
            annotation_position="top right"
        )

        fig_trend.update_layout(
            height=220,
            margin=dict(l=10, r=80, t=15, b=10),
            paper_bgcolor="#0f0f18",
            plot_bgcolor="#0f0f18",
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color="#888", size=10),
                tickangle=-30,
                showline=True,
                linecolor="#333"
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#1e1e2e",
                tickfont=dict(color="#888"),
                title="Failures",
                titlefont=dict(color="#666"),
                rangemode="tozero"
            ),
            font=dict(color="#ccc"),
            showlegend=False
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

        # Summary row
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak Failures", max(trend), delta=f"Cycle {trend_labels[peak_idx]}", delta_color="inverse")
        c2.metric("Avg per Cycle", f"{avg:.1f}")
        c3.metric("Latest Cycle", trend[-1], delta=str(trend[-1] - trend[-2]) if len(trend) > 1 else "—", delta_color="inverse")