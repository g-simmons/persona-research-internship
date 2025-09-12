import pandas as pd
import json
import altair as alt

# ----------------- Load & merge -----------------
scores = pd.read_csv("llm_ontology/data/term_scores_all.csv")      # per-term scores
with open("llm_ontology/data/term_depths.json") as f:
    depth_map = json.load(f)

depth_df = pd.DataFrame(depth_map.items(), columns=["term", "depth"])
data     = scores.merge(depth_df, on="term", how="inner")

# Keep just the three model sizes you care about
model_order = ["70M", "160M", "1.4B"]
data        = data[data["model"].isin(model_order)]

# ── Long format ──────────────────────────────────────────────────────────
metric_labels = {
    "causal_sep_score": "Causal-sep",
    "hierarchy_score":  "Hierarchy",
    "linear_rep_score": "Linear-rep",
}

long = (
    data.melt(
        id_vars=["term", "depth", "step", "model"],
        value_vars=list(metric_labels.keys()),
        var_name="metric",
        value_name="score",
    )
)
long["metric"] = long["metric"].map(metric_labels)

# ── Slider parameter (Altair-5) ──────────────────────────────────────────
step_min, step_max = int(long["step"].min()), int(long["step"].max())

step_param = alt.param(
    name="step",
    bind=alt.binding_range(min=step_min, max=step_max, step=2_000, name="Training Step:"),
    value=step_min,
)

# ── Scatter layer ───────────────────────────────────────────────────────
scatter = (
    alt.Chart(long)
        .add_params(step_param)
        .transform_filter(alt.datum.step == step_param)
        .mark_circle(size=40, opacity=0.35)
        .encode(
            x=alt.X("depth:Q", title="Ontology depth"),
            y=alt.Y("score:Q", title="Term score"),
            color=alt.Color("model:N", legend=None),
            tooltip=["term", "depth", "score", "model", "step"],
        )
        .properties(width=180, height=140)
)

# ── Trend-line layer (per-model, per-metric) ────────────────────────────
trend = (
    alt.Chart(long)
      .add_params(step_param)
      .transform_filter(alt.datum.step == step_param)
      .transform_loess(              # ← use LOESS helper
          "depth", "score",
          groupby=["model", "metric"],
          bandwidth=0.5              # tweak for more / less smoothing
      )
      .mark_line(size=2)
      .encode(
          x="depth:Q",
          y="score:Q",
          color=alt.Color("model:N", legend=None),
      )
)

# ── Facet & layer ───────────────────────────────────────────────────────
chart = (
    (scatter + trend)                           # layer scatter & line
    .facet(
        row=alt.Row("metric:N",
                    sort=["Causal-sep", "Hierarchy", "Linear-rep"],
                    header=alt.Header(labelAngle=0)),
        column=alt.Column("model:N",
                          sort=model_order,
                          header=alt.Header(labelAngle=0)),
        spacing=8,
    )
    .resolve_scale(y="independent")             # independent y per metric row
    .properties(title="Depth vs Score per Model (slider → training step)")
)

chart.save("depth_score_3x3_with_trend.html")
print("Saved → depth_score_3x3_with_trend.html")