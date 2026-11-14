"""Tab 2: Frequentist A/B, CUPED Variance Reduction, Sequential & Bayesian Lab."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.experimentation.bayesian_ab import BayesianABResult, run_bayesian_conversion_test
from src.experimentation.frequentist_ab import (
    CUPEDComparisonResult,
    FrequentistTestResult,
    analyze_experiment_from_duckdb,
)
from src.experimentation.power_analysis import calculate_proportion_sample_size, generate_power_curves
from src.experimentation.sequential_testing import SequentialTestResult, run_sequential_from_daily_logs


@st.cache_data(ttl=300, max_entries=6, show_spinner=False)
def _cached_ab_duckdb(alpha: float) -> tuple[FrequentistTestResult, CUPEDComparisonResult]:
    return analyze_experiment_from_duckdb(alpha=alpha)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def _cached_sequential(metric: str, alpha: float) -> SequentialTestResult:
    return run_sequential_from_daily_logs(metric=metric, alpha=alpha)


def render_ab_testing_tab() -> dict[str, Any]:
    """Renders Tab 2 and returns A/B metrics payload for reporting."""
    c_ctrl1, c_ctrl2 = st.columns([1.0, 1.5])
    with c_ctrl1:
        alpha = st.select_slider("Significance Level (alpha)", options=[0.01, 0.05, 0.10], value=0.05)
    with c_ctrl2:
        rope_bps = st.slider("Bayesian Conversion ROPE Half-Width (bps)", 5, 60, 20, step=5)

    conv_res, cuped_res = _cached_ab_duckdb(alpha)
    c_conv = round(conv_res.control_mean * conv_res.n_control)
    t_conv = round(conv_res.treatment_mean * conv_res.n_treatment)
    bayes_res: BayesianABResult = run_bayesian_conversion_test(
        c_conv,
        conv_res.n_control,
        t_conv,
        conv_res.n_treatment,
        rope=(-rope_bps / 10_000.0, rope_bps / 10_000.0),
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Conv. Relative Lift", f"{conv_res.relative_lift_pct:+.2f}%", f"p={conv_res.p_value:.5f}")
    k2.metric(
        "CUPED Revenue Lift",
        f"${cuped_res.cuped.absolute_lift:.3f}/user",
        f"p={cuped_res.cuped.p_value:.5f}",
    )
    k3.metric("CUPED Variance Reduction", f"{cuped_res.variance_reduction_pct:.1f}%", f"theta={cuped_res.theta:.3f}")
    k4.metric("Bayesian P(T > C)", f"{bayes_res.prob_treatment_beats_control * 100:.2f}%")
    k5.metric("Expected Loss (Ship T)", f"{bayes_res.expected_loss_treatment:.6f}")

    st.subheader("Unadjusted Welch t-Test vs. CUPED Covariate Adjustment")
    comp_df = pd.DataFrame(
        [
            {
                "Estimator": "Unadjusted Post-Exp Revenue",
                "Control Mean ($)": round(cuped_res.unadjusted.control_mean, 4),
                "Treatment Mean ($)": round(cuped_res.unadjusted.treatment_mean, 4),
                "Lift ($)": round(cuped_res.unadjusted.absolute_lift, 4),
                "Std Error ($)": round(cuped_res.unadjusted.standard_error, 4),
                "t-Statistic": round(cuped_res.unadjusted.test_statistic, 3),
                "p-Value": round(cuped_res.unadjusted.p_value, 6),
                "SE Reduction (%)": 0.0,
            },
            {
                "Estimator": "CUPED-Adjusted Revenue",
                "Control Mean ($)": round(cuped_res.cuped.control_mean, 4),
                "Treatment Mean ($)": round(cuped_res.cuped.treatment_mean, 4),
                "Lift ($)": round(cuped_res.cuped.absolute_lift, 4),
                "Std Error ($)": round(cuped_res.cuped.standard_error, 4),
                "t-Statistic": round(cuped_res.cuped.test_statistic, 3),
                "p-Value": round(cuped_res.cuped.p_value, 6),
                "SE Reduction (%)": round(cuped_res.se_reduction_pct, 2),
            },
        ]
    )
    st.dataframe(comp_df, width="stretch", hide_index=True)

    col_seq, col_bay = st.columns(2)
    with col_seq:
        st.subheader("O'Brien-Fleming Group-Sequential Monitoring")
        seq_metric = st.radio(
            "Sequential Metric",
            ["post_exp_revenue", "converted"],
            horizontal=True,
            key="tab2_seq_metric",
        )
        seq_res = _cached_sequential(seq_metric, alpha)
        sched = seq_res.schedule
        fig_seq = go.Figure()
        fig_seq.add_trace(
            go.Scatter(
                x=sched["information_fraction"],
                y=sched["efficacy_boundary_z"],
                name="Upper OBF Boundary",
                line={"color": "#DC2626", "dash": "dash"},
            )
        )
        fig_seq.add_trace(
            go.Scatter(
                x=sched["information_fraction"],
                y=sched["neg_efficacy_boundary_z"],
                name="Lower OBF Boundary",
                line={"color": "#DC2626", "dash": "dash"},
            )
        )
        fig_seq.add_trace(
            go.Scatter(
                x=sched["information_fraction"],
                y=sched["observed_z"],
                mode="lines+markers",
                name="Observed Cumulative Z",
                line={"color": "#1B365D", "width": 3},
            )
        )
        fig_seq.update_layout(
            xaxis_title="Information Fraction (t_k)",
            yaxis_title="Z-Statistic",
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_seq, width="stretch")
        st.caption(
            f"Decision: `{seq_res.decision}` at Look {seq_res.stop_look}/{seq_res.k_looks} "
            f"(t={seq_res.information_fraction_at_stop:.2f}, final Z={seq_res.final_z_stat:.2f})."
        )

    with col_bay:
        st.subheader("Analytical Beta-Binomial Posterior PDFs")
        pp = bayes_res.posterior_params
        x_lo = min(bayes_res.control_hdi_95[0], bayes_res.treatment_hdi_95[0]) * 0.96
        x_hi = max(bayes_res.control_hdi_95[1], bayes_res.treatment_hdi_95[1]) * 1.04
        grid = np.linspace(x_lo, x_hi, 300)
        pdf_c = stats.beta.pdf(grid, pp["alpha_c"], pp["beta_c"])
        pdf_t = stats.beta.pdf(grid, pp["alpha_t"], pp["beta_t"])

        fig_bay = go.Figure()
        fig_bay.add_trace(go.Scatter(x=grid * 100, y=pdf_c, name="Control Posterior", fill="tozeroy"))
        fig_bay.add_trace(go.Scatter(x=grid * 100, y=pdf_t, name="Treatment Posterior", fill="tozeroy"))
        fig_bay.update_layout(
            xaxis_title="Conversion Rate (%)",
            yaxis_title="Posterior Density",
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_bay, width="stretch")
        st.caption(
            f"95% Lift HDI: [{bayes_res.lift_hdi_95[0] * 100:.3f}%, {bayes_res.lift_hdi_95[1] * 100:.3f}%] | "
            f"P(in ROPE) = {bayes_res.prob_in_rope * 100:.2f}%"
        )

    with st.expander("Statistical Power & Minimum Detectable Effect (MDE) Calculator", expanded=False):
        p1, p2, p3 = st.columns(3)
        base_rate_pct = p1.slider("Baseline CVR (%)", 2.0, 25.0, 12.0, 0.5, key="pwr_base")
        rel_mde_pct = p2.slider("Target Relative MDE (%)", 1.0, 20.0, 5.0, 0.5, key="pwr_mde")
        target_power = p3.slider("Target Power (1 - beta)", 0.70, 0.95, 0.80, 0.05, key="pwr_tgt")

        req = calculate_proportion_sample_size(
            base_rate_pct / 100.0,
            rel_mde_pct / 100.0,
            alpha=alpha,
            power=target_power,
            is_relative_mde=True,
        )
        st.info(
            f"Required Sample Size: **{req.n_per_arm:,} users/arm** (**{req.total_sample_size:,} total**) "
            f"to detect +{rel_mde_pct:.1f}% relative lift at {target_power * 100:.0f}% power."
        )
        base_abs = base_rate_pct / 100.0
        mde_grid = [
            rel_mde_pct * 0.6 / 100.0 * base_abs,
            req.absolute_mde,
            rel_mde_pct * 1.4 / 100.0 * base_abs,
        ]
        curves_df = generate_power_curves(base_abs, mde_grid, alpha=alpha)
        fig_pwr = go.Figure()
        for mde_val, sub in curves_df.groupby("absolute_mde"):
            fig_pwr.add_trace(go.Scatter(x=sub["n_per_arm"], y=sub["power"], name=f"Abs MDE={mde_val:.4f}"))
        fig_pwr.add_hline(y=target_power, line_dash="dot", line_color="gray")
        fig_pwr.update_layout(height=280, xaxis_title="Users per Arm", yaxis_title="Statistical Power")
        st.plotly_chart(fig_pwr, width="stretch")

    return {
        "n_control": conv_res.n_control,
        "n_treatment": conv_res.n_treatment,
        "conv_control": c_conv,
        "conv_treatment": t_conv,
        "cuped_var_reduction_pct": cuped_res.variance_reduction_pct,
        "ctrl_rev_user": cuped_res.cuped.control_mean,
        "treat_rev_user": cuped_res.cuped.treatment_mean,
        "ab_revenue_lift": cuped_res.cuped.absolute_lift,
        "bayesian_prob_win": bayes_res.prob_treatment_beats_control,
        "bayesian_expected_loss": bayes_res.expected_loss_treatment,
        "daily_traffic": round((conv_res.n_control + conv_res.n_treatment) / 14.0, 0),
    }
