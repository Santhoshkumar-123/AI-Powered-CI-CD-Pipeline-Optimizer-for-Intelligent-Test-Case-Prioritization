import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def render_ai_recommendations(api_client, limit: int = 20):
    """Render AI-powered insights with charts and risk breakdown"""
    st.subheader("🤖 AI-Recommended Test Order")

    try:
        recommendations = api_client.get_ai_recommendations(limit=limit)

        if not recommendations:
            st.info("No recommendations available. Run simulation first.")
            return

        df = pd.DataFrame(recommendations)

        # ── Summary metrics ──────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        high_risk   = len(df[df['failure_probability'] > 0.7])
        medium_risk = len(df[(df['failure_probability'] > 0.4) & (df['failure_probability'] <= 0.7)])
        low_risk    = len(df[df['failure_probability'] <= 0.4])
        avg_score   = df['ai_score'].mean() if 'ai_score' in df.columns else 0

        with col1:
            st.metric("Tests Analysed", len(df))
        with col2:
            st.metric("🔴 High Risk", high_risk)
        with col3:
            st.metric("🟡 Medium Risk", medium_risk)
        with col4:
            st.metric("Avg AI Score", f"{avg_score:.3f}")

        st.markdown("---")

        # ── Risk distribution bar chart ───────────────────────────────────────
        col_chart, col_scatter = st.columns(2)

        with col_chart:
            st.markdown("##### Risk Distribution")
            fig_risk = go.Figure(go.Bar(
                x=["High Risk", "Medium Risk", "Low Risk"],
                y=[high_risk, medium_risk, low_risk],
                marker_color=["#EF4444", "#F59E0B", "#10B981"],
                text=[high_risk, medium_risk, low_risk],
                textposition="outside"
            ))
            fig_risk.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Test Count",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with col_scatter:
            st.markdown("##### AI Score vs Failure Probability")
            if 'failure_probability' in df.columns and 'ai_score' in df.columns:
                # Fetch more data for a richer scatter
                all_recs = api_client.get_ai_recommendations(limit=100)
                df_full = pd.DataFrame(all_recs) if all_recs else df

                # Add jitter to separate overlapping points
                rng = np.random.default_rng(42)
                df_full = df_full.copy()
                df_full['fp_jitter'] = df_full['failure_probability'] + rng.uniform(-0.02, 0.02, len(df_full))
                df_full['ai_jitter'] = df_full['ai_score'] + rng.uniform(-0.02, 0.02, len(df_full))

                color_map = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}

                fig_scatter = go.Figure()
                for priority_val, color in color_map.items():
                    subset = df_full[df_full['priority'] == priority_val] if 'priority' in df_full.columns else df_full
                    if len(subset) == 0:
                        continue
                    fig_scatter.add_trace(go.Scatter(
                        x=subset['fp_jitter'],
                        y=subset['ai_jitter'],
                        mode='markers',
                        name=priority_val.capitalize(),
                        marker=dict(
                            color=color,
                            size=10,
                            opacity=0.8,
                            line=dict(color='rgba(255,255,255,0.3)', width=1)
                        ),
                        text=subset['test_id'] if 'test_id' in subset.columns else None,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Fail Prob: %{x:.3f}<br>"
                            "AI Score: %{y:.3f}<extra></extra>"
                        )
                    ))

                # Diagonal reference line (perfect correlation)
                fig_scatter.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.15)', dash='dot', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                fig_scatter.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="#0f0f18",
                    plot_bgcolor="#0f0f18",
                    xaxis=dict(
                        title="Failure Probability",
                        range=[-0.05, 1.05],
                        showgrid=True, gridcolor="#1e1e2e",
                        tickfont=dict(color="#888"),
                        titlefont=dict(color="#aaa")
                    ),
                    yaxis=dict(
                        title="AI Score",
                        range=[-0.05, 1.05],
                        showgrid=True, gridcolor="#1e1e2e",
                        tickfont=dict(color="#888"),
                        titlefont=dict(color="#aaa")
                    ),
                    legend=dict(
                        font=dict(color="#ccc", size=11),
                        bgcolor="rgba(0,0,0,0)",
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right", x=1
                    ),
                    font=dict(color="#ccc")
                )
                st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # ── Ranked test table with colour-coded priority + reason ─────────────
        st.markdown("##### Top Prioritized Tests")

        # Fetch full prioritized list (includes reason field from backend)
        try:
            prioritized_data = api_client.get_prioritized_tests()
            p_list = prioritized_data.get("prioritized", []) if prioritized_data else []
            if p_list:
                df_table = pd.DataFrame(p_list)
            else:
                df_table = df.copy()
        except Exception:
            df_table = df.copy()

        # Build reason column if not present — derive from available fields
        if 'reason' not in df_table.columns:
            def _derive_reason(row):
                parts = []
                fp = row.get('failure_probability', 0)
                score = row.get('ai_score', 0)
                if fp == 1.0:
                    parts.append("Failed in current cycle")
                elif fp > 0.7:
                    parts.append(f"High failure probability ({fp:.2f})")
                if score > 0.9:
                    parts.append(f"Top AI score ({score:.3f})")
                if not parts:
                    parts.append(f"AI score {score:.3f} above threshold")
                return " | ".join(parts)
            df_table['reason'] = df_table.apply(_derive_reason, axis=1)

        display_cols = [c for c in
            ['test_id', 'ai_score', 'failure_probability', 'execution_time', 'priority', 'reason']
            if c in df_table.columns]

        def _colour_priority(val):
            colours = {"high": "color: #EF4444; font-weight:bold",
                       "medium": "color: #F59E0B",
                       "low": "color: #10B981"}
            return colours.get(val, "")

        fmt = {}
        if 'ai_score' in display_cols:           fmt['ai_score'] = "{:.4f}"
        if 'failure_probability' in display_cols: fmt['failure_probability'] = "{:.4f}"
        if 'execution_time' in display_cols:      fmt['execution_time'] = "{:.3f}"

        styled = df_table[display_cols].head(limit).style
        if 'priority' in display_cols:
            styled = styled.applymap(_colour_priority, subset=["priority"])
        styled = styled.format(fmt)

        st.dataframe(styled, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading recommendations: {str(e)}")

def render_high_risk_tests(api_client, threshold: float = 0.8):
    """Render tests with high failure probability"""
    st.subheader("⚠️ High Risk Tests")

    try:
        risky_tests = api_client.get_high_risk_tests(threshold=threshold)

        if not risky_tests:
            st.success("✅ No high-risk tests detected above this threshold")
            return

        df = pd.DataFrame(risky_tests)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 High Risk Tests", len(df))
        col2.metric("Avg AI Score", f"{df['ai_score'].mean():.3f}" if 'ai_score' in df.columns else "—")
        col3.metric("Max Failure Prob", f"{df['failure_probability'].max():.1%}" if 'failure_probability' in df.columns else "—")

        st.markdown("---")

        # Bar chart: AI score per test
        df_sorted = df.sort_values('ai_score', ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=df_sorted['ai_score'],
            y=df_sorted['test_id'],
            orientation='h',
            marker=dict(
                color=df_sorted['ai_score'],
                colorscale=[[0, "#F59E0B"], [0.5, "#EF4444"], [1, "#7F1D1D"]],
                showscale=False,
                line=dict(width=0)
            ),
            text=[f"{v:.3f}" for v in df_sorted['ai_score']],
            textposition="outside",
            textfont=dict(color="#ccc", size=10),
            hovertemplate="<b>%{y}</b><br>AI Score: %{x:.4f}<extra></extra>"
        ))
        fig.update_layout(
            height=max(250, len(df_sorted) * 22),
            margin=dict(l=0, r=50, t=10, b=10),
            paper_bgcolor="#0f0f18",
            plot_bgcolor="#0f0f18",
            xaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#222",
                       tickfont=dict(color="#888"), title="AI Risk Score"),
            yaxis=dict(tickfont=dict(color="#ccc"), showgrid=False),
            font=dict(color="#ccc")
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Detail table
        preferred = ['test_id', 'ai_score', 'failure_probability', 'last_failure',
                     'failure_count', 'execution_time', 'drift_phase']
        available = [c for c in preferred if c in df.columns]
        st.dataframe(
            df[available].sort_values('ai_score', ascending=False).head(20),
            use_container_width=True, hide_index=True
        )

    except Exception as e:
        st.error(f"Error loading high-risk tests: {str(e)}")

def render_test_search(api_client):
    """Render test search functionality"""
    st.subheader("🔍 Search Tests")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by Test ID or keyword",
            placeholder="e.g., REG_0001 or login",
            key="test_search"
        )
    
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    if search_button and search_query:
        try:
            results = api_client.search_tests(search_query)
            
            if results:
                st.success(f"Found {len(results)} test(s)")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No tests found matching your query")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

def render_test_details(api_client, test_id: str):
    """Render detailed test information"""
    try:
        test_info = api_client.get_test_details(test_id)
        
        if test_info:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Success Rate", f"{test_info.get('success_rate', 0)}%")
            with col2:
                st.metric("Avg Duration", f"{test_info.get('avg_duration', 0):.2f}s")
            with col3:
                st.metric("Total Runs", test_info.get('total_runs', 0))
            with col4:
                st.metric("Failures", test_info.get('failure_count', 0))
            
            # Historical performance
            if 'history' in test_info and test_info['history']:
                st.subheader("Historical Performance")
                history_df = pd.DataFrame(test_info['history'])
                fig = px.line(history_df, x='date', y='duration', title='Execution Time Trend')
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent failures
            if 'recent_failures' in test_info and test_info['recent_failures']:
                st.subheader("Recent Failures")
                failures_df = pd.DataFrame(test_info['recent_failures'])
                st.dataframe(failures_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Test details not available")
    except Exception as e:
        st.error(f"Error loading test details: {str(e)}")

def render_historical_trends(api_client, start_date, end_date):
    """Render historical trend analysis"""
    st.subheader("📊 Historical Trends")
    
    try:
        historical_data = api_client.get_historical_metrics(
            start_date.isoformat(),
            end_date.isoformat()
        )
        
        if historical_data and 'metrics' in historical_data:
            metrics = historical_data['metrics']
            
            # Create trend chart
            fig = go.Figure()
            
            if 'dates' in metrics and 'pass_rates' in metrics:
                fig.add_trace(go.Scatter(
                    x=metrics['dates'],
                    y=metrics['pass_rates'],
                    name='Pass Rate',
                    line=dict(color='green', width=2)
                ))
            
            if 'dates' in metrics and 'failure_rates' in metrics:
                fig.add_trace(go.Scatter(
                    x=metrics['dates'],
                    y=metrics['failure_rates'],
                    name='Failure Rate',
                    line=dict(color='red', width=2)
                ))
            
            fig.update_layout(
                title='Test Results Over Time',
                xaxis_title='Date',
                yaxis_title='Percentage',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show improvement metrics
            if 'improvement' in historical_data:
                improvement = historical_data['improvement']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pass Rate Change", f"{improvement.get('pass_rate_change', 0)}%")
                with col2:
                    st.metric("Avg Duration Change", f"{improvement.get('duration_change', 0)}%")
                with col3:
                    st.metric("Failure Reduction", f"{improvement.get('failure_reduction', 0)}%")
        else:
            st.info("No historical data available for selected date range")
    except Exception as e:
        st.error(f"Error loading historical trends: {str(e)}")

def render_export_options(api_client, filters: Dict):
    """Render export functionality — overall summary report"""
    st.subheader("📥 Export Overall Report")
    st.caption("Downloads a structured summary report including metrics, baseline comparison, drift events, and training loss — not raw data.")

    # Preview what's included
    with st.expander("📋 What's included in the report"):
        st.markdown("""
        - **Overall Metrics** — total tests, pass/fail rate, avg execution time
        - **Per-Phase Summary** — breakdown by Backend, Frontend, Database, API, Mobile
        - **Baseline Comparison** — Mean APFD, NAPFD, TTFF for all 6 methods
        - **Drift Events** — cycles where concept drift was detected + LR changes
        - **Training Loss Summary** — initial, final, and best loss across epochs
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        export_format = st.selectbox("Select Format", ["CSV", "JSON"], key="export_format")
    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("⬇️ Download Report", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating report..."):
                    report_data = api_client.export_report(export_format.lower(), {})
                if report_data:
                    ext  = export_format.lower()
                    mime = "text/csv" if ext == "csv" else "application/json"
                    fname = f"semanti_q_report.{ext}"
                    st.download_button(
                        label=f"💾 Save {export_format} Report",
                        data=report_data,
                        file_name=fname,
                        mime=mime,
                        use_container_width=True
                    )
                    st.success("Report ready — click above to save.")
                else:
                    st.error("Failed to generate report")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

def render_alert_configuration(api_client):
    """Render alert configuration panel with real email support"""
    st.subheader("🔔 Alert Configuration")

    try:
        current = api_client.get_alert_settings() or {}

        col1, col2 = st.columns(2)

        with col1:
            failure_threshold = st.slider(
                "Alert if failure rate exceeds (%)", 0, 100,
                current.get('failure_threshold', 10), key="failure_threshold"
            )
            execution_threshold = st.slider(
                "Alert if execution time exceeds (min)", 0, 120,
                current.get('execution_threshold', 30), key="execution_threshold"
            )

        with col2:
            alert_channels = st.multiselect(
                "Alert Channels", ["Email", "Slack", "Teams", "SMS"],
                default=current.get('channels', ["Email"]), key="alert_channels"
            )
            alert_frequency = st.selectbox(
                "Alert Frequency", ["Immediate", "Hourly", "Daily"],
                index=["Immediate", "Hourly", "Daily"].index(
                    current.get('frequency', 'Immediate')),
                key="alert_frequency"
            )

        # Email credentials section
        if "Email" in alert_channels:
            st.markdown("---")
            st.markdown("##### 📧 Email Configuration")
            st.caption("Uses Gmail SMTP. For sender, create an App Password at myaccount.google.com → Security → App Passwords")

            ec1, ec2 = st.columns(2)
            with ec1:
                recipient_email = st.text_input(
                    "Recipient Email",
                    value=current.get('recipient_email', 'santhosh5375kumar@gmail.com'),
                    key="recipient_email"
                )
                sender_email = st.text_input(
                    "Sender Gmail Address",
                    value=current.get('sender_email', ''),
                    placeholder="yourname@gmail.com",
                    key="sender_email"
                )
            with ec2:
                sender_password = st.text_input(
                    "Gmail App Password",
                    value="",
                    type="password",
                    placeholder="16-char app password",
                    key="sender_password",
                    help="Go to Google Account → Security → 2-Step Verification → App Passwords"
                )
                st.markdown("")
                st.markdown("")
                if st.button("📨 Send Test Email", use_container_width=True):
                    if sender_email and sender_password:
                        with st.spinner("Sending test email..."):
                            result = api_client.trigger_alert({
                                "title": "Test Alert",
                                "recipient_email": recipient_email,
                                "sender_email": sender_email,
                                "sender_password": sender_password,
                                "body": (
                                    f"This is a test alert from Semanti-Q.\n\n"
                                    f"Failure threshold : {failure_threshold}%\n"
                                    f"Exec threshold    : {execution_threshold} min\n"
                                    f"Frequency         : {alert_frequency}"
                                )
                            })
                        if result and result.get('status') == 'sent':
                            st.success(f"✅ Test email sent to {recipient_email}")
                        else:
                            st.error("❌ Failed — check credentials")
                    else:
                        st.warning("Enter sender email and app password first")
        else:
            recipient_email = current.get('recipient_email', '')
            sender_email    = current.get('sender_email', '')
            sender_password = ''

        st.markdown("---")
        if st.button("💾 Save Alert Settings", type="primary"):
            config = {
                'failure_threshold': failure_threshold,
                'execution_threshold': execution_threshold,
                'channels': alert_channels,
                'frequency': alert_frequency,
                'recipient_email': recipient_email,
                'sender_email': sender_email,
            }
            if sender_password:
                config['sender_password'] = sender_password

            result = api_client.configure_alerts(config)
            if result:
                if result.get('test_email_sent'):
                    st.success(f"✅ Settings saved — confirmation email sent to {recipient_email}")
                else:
                    st.success("✅ Settings saved (add sender credentials to enable email)")
            else:
                st.error("Failed to save settings")

    except Exception as e:
        st.error(f"Error configuring alerts: {str(e)}")

def render_test_suite_management(api_client):
    """Render test suite management"""
    st.subheader("🧪 Test Suite Management")
    
    tab1, tab2 = st.tabs(["Create Suite", "Run Suite"])
    
    with tab1:
        suite_name = st.text_input("Suite Name", placeholder="e.g., Critical Regression Tests")
        
        # Get available tests (mock for now)
        all_tests = [f"REG_{i:04d}" for i in range(100)]
        selected_tests = st.multiselect("Select Tests", all_tests, key="suite_tests")
        
        if st.button("Create Suite", type="primary"):
            if suite_name and selected_tests:
                result = api_client.create_test_suite(suite_name, selected_tests)
                if result:
                    st.success(f"✅ Suite '{suite_name}' created with {len(selected_tests)} tests")
                else:
                    st.error("Failed to create suite")
            else:
                st.warning("Please provide suite name and select tests")
    
    with tab2:
        try:
            suites = api_client.get_test_suites()
            
            if suites:
                suite_names = [s['name'] for s in suites]
                selected_suite = st.selectbox("Select Suite", suite_names)
                
                if st.button("▶️ Run Suite", type="primary"):
                    suite_id = next(s['id'] for s in suites if s['name'] == selected_suite)
                    
                    with st.spinner("Running test suite..."):
                        result = api_client.run_test_suite(suite_id)
                        
                        if result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Tests", result.get('total', 0))
                            with col2:
                                st.metric("Passed", result.get('passed', 0))
                            with col3:
                                st.metric("Failed", result.get('failed', 0))
                            
                            if result.get('passed', 0) == result.get('total', 0):
                                st.success("✅ All tests passed!")
                            else:
                                st.warning(f"⚠️ {result.get('failed', 0)} test(s) failed")
            else:
                st.info("No test suites available. Create one first.")
        except Exception as e:
            st.error(f"Error managing test suites: {str(e)}")

def render_comparison_view(api_client, build_cycles):
    """Render side-by-side build comparison"""
    st.subheader("🔄 Build Comparison")

    col1, col2 = st.columns(2)
    with col1:
        build1 = st.selectbox("Select Build 1", build_cycles, index=0, key="build1")
    with col2:
        build2 = st.selectbox("Select Build 2", build_cycles,
                               index=min(1, len(build_cycles)-1), key="build2")

    metrics1 = api_client.get_dashboard_metrics(build_cycle=build1) or {}
    metrics2 = api_client.get_dashboard_metrics(build_cycle=build2) or {}

    if not metrics1 or not metrics2:
        st.warning("Could not load metrics for selected builds.")
        return

    # ── Side-by-side metric cards ─────────────────────────────────────────────
    st.markdown("---")
    headers = st.columns([2, 1, 1])
    headers[0].markdown("**Metric**")
    headers[1].markdown(f"**{build1}**")
    headers[2].markdown(f"**{build2}**")

    def _delta_color(val):
        if val > 0: return "#10B981"
        if val < 0: return "#EF4444"
        return "#888"

    rows = [
        ("Total Tests",      metrics1.get('total_tests', 0),      metrics2.get('total_tests', 0),      False),
        ("Pass Rate (%)",    metrics1.get('pass_rate', 0),         metrics2.get('pass_rate', 0),         True),
        ("Failures",         metrics1.get('critical_failures', 0), metrics2.get('critical_failures', 0), False),
        ("Tests Prioritized",metrics1.get('tests_prioritized', 0), metrics2.get('tests_prioritized', 0), True),
        ("Time Saved (min)", metrics1.get('time_saved_minutes', 0),metrics2.get('time_saved_minutes', 0),True),
    ]

    for label, v1, v2, higher_is_better in rows:
        diff = v2 - v1 if isinstance(v2, (int, float)) and isinstance(v1, (int, float)) else 0
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "—")
        color = _delta_color(diff if higher_is_better else -diff)
        r = st.columns([2, 1, 1])
        r[0].markdown(f"<span style='color:#aaa'>{label}</span>", unsafe_allow_html=True)
        r[1].markdown(f"<b>{v1:,.2f}" if isinstance(v1, float) else f"<b>{v1:,}</b>", unsafe_allow_html=True)
        r[2].markdown(
            f"<b>{v2:,.2f}</b> <span style='color:{color};font-size:12px'>{arrow} {abs(diff):.2f}</span>"
            if isinstance(v2, float) else
            f"<b>{v2:,}</b> <span style='color:{color};font-size:12px'>{arrow} {abs(diff)}</span>",
            unsafe_allow_html=True
        )

    # ── Visual bar comparison ─────────────────────────────────────────────────
    st.markdown("---")
    metrics_to_plot = ["total_tests", "critical_failures", "tests_prioritized"]
    labels_map = {"total_tests": "Total Tests", "critical_failures": "Failures", "tests_prioritized": "Prioritized"}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=build1,
        x=[labels_map[m] for m in metrics_to_plot],
        y=[metrics1.get(m, 0) for m in metrics_to_plot],
        marker_color="#3B82F6",
        text=[metrics1.get(m, 0) for m in metrics_to_plot],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name=build2,
        x=[labels_map[m] for m in metrics_to_plot],
        y=[metrics2.get(m, 0) for m in metrics_to_plot],
        marker_color="#10B981",
        text=[metrics2.get(m, 0) for m in metrics_to_plot],
        textposition="outside"
    ))
    fig.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#0f0f18",
        plot_bgcolor="#0f0f18",
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickfont=dict(color="#ccc"), showgrid=False),
        yaxis=dict(tickfont=dict(color="#888"), gridcolor="#222"),
        font=dict(color="#ccc")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Pass rate comparison gauge-style
    pr1 = metrics1.get('pass_rate', 0)
    pr2 = metrics2.get('pass_rate', 0)
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"Pass Rate — {build1}", f"{pr1:.2f}%")
        st.progress(int(pr1))
    with c2:
        st.metric(f"Pass Rate — {build2}", f"{pr2:.2f}%",
                  delta=f"{pr2 - pr1:+.2f}%")
        st.progress(int(pr2))
