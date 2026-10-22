"""Explain the Model, Gator football edition (unofficial fan theme).

Three regression models taught through the cycles of a football season.
Every number comes from outputs/<dataset>.json or the m10 helpers. Nothing is retrained or invented.

Run locally:  streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import m10

LABELS = {"glucose": "Fasting glucose", "sbp": "Blood pressure", "fev1": "Lung function (FEV₁)"}
ORANGE, BLUE, NAVY, GREEN, GRAY, LIGHT = "#FA4616", "#0021A5", "#1B1F3B", "#2E7D32", "#6c757d", "#C3C9D9"

CSS = f"""
<style>
h1 {{ color: {BLUE}; }}
h3 {{ color: {NAVY}; border-bottom: 3px solid {ORANGE}; padding-bottom: 4px; margin-top: 1.6rem; }}
.scoreboard {{ background: {NAVY}; color: white; border-radius: 10px; padding: 14px 18px;
  font-family: ui-monospace, Menlo, monospace; font-size: 1.05rem; line-height: 1.7; }}
.scoreboard b {{ color: {ORANGE}; }}
.field {{ background: repeating-linear-gradient(90deg, #2E7D32 0 46px, #33873a 46px 92px);
  color: white; border-radius: 10px; padding: 12px 18px; font-weight: 600; }}
</style>
"""


# ---- shared helpers ----------------------------------------------------------
def fmt(v: float, u: str, d: int = 1) -> str:
    return f"{v:,.{d}f} {u}".strip()


def signed(v: float, u: str = "", d: int = 1) -> str:
    return f"{v:+,.{d}f} {u}".strip()


def pick_person() -> tuple[dict, dict]:
    st.sidebar.markdown("### 🏈 Pick your matchup")
    key = st.sidebar.selectbox("Dataset", m10.DATASETS, format_func=LABELS.get, key="dataset")
    b = m10.load_bundle(key)
    labels = {p["id"]: f"{p['label']}: {p['description']}" for p in b["people"]}
    pid = st.sidebar.selectbox("Example person", list(labels), format_func=labels.get, key=f"person_{key}")
    person = next(p for p in b["people"] if p["id"] == pid)
    st.sidebar.caption(
        f"**Predicting:** {b['target_label']} ({b['units']}).\n\n"
        "The five example people are real participants in the CDC's NHANES health survey, from the "
        "validation group (people the models never trained on). The football story is only an analogy; "
        "every number is real model output.")
    st.sidebar.caption("Unofficial fan-themed class project. Not affiliated with the University of Florida.")
    return b, person


def kickoff_header(title: str, subtitle: str, big_idea: str, b: dict, person: dict) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(f"🐊 {title}")
    st.caption(subtitle)
    st.success(f"**The big idea:** {big_idea}")
    st.markdown(
        f"<div class='field'>Today's matchup: {LABELS[b['id']]} · {person['label']} "
        f"({person['description']}) · actually measured: {fmt(person['observed'], b['units'])}</div>",
        unsafe_allow_html=True)


def phase(icon: str, name: str, topic: str) -> None:
    st.markdown(f"### {icon} {name} · {topic}")


def gator_tip(text: str) -> None:
    st.info(text, icon="🐊")


def label_of(b: dict, name: str) -> str:
    groups = {g["name"]: g["label"] for g in b["categorical_groups"]}
    if name in groups:
        return groups[name]
    return next((f["label"] for f in b["features"] if f["name"] == name), name)


def short_label(b: dict, name: str) -> str:
    return label_of(b, name).split(" (")[0]


def feat_units(b: dict, name: str) -> str:
    return next((f["units"] for f in b["features"] if f["name"] == name), "")


def chart(fig: go.Figure, height: int = 380) -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=60), plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig.update_xaxes(gridcolor="#eceff5")
    fig.update_yaxes(gridcolor="#eceff5")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# =============================================================================
# 1. PUSH & PULL (linear regression) · THE GAME PLAN
# =============================================================================
def linear_page() -> None:
    b, person = pick_person()
    u, lin, groups = b["units"], b["linear"], b["categorical_groups"]
    grouped_cols = {lv["column"] for g in groups for lv in g["levels"] if lv["column"]}
    numeric = [f for f in b["features"] if f["name"] not in grouped_cols]

    kickoff_header("Push & Pull: The Game Plan",
                   "Linear regression: how one person's data becomes a prediction",
                   "The model starts from an average prediction and adds one push per feature, up or "
                   "down. Starting point + all pushes = the prediction.", b, person)

    # --- call your own plays (edit) ---
    wkey = f"lin_{b['id']}_{person['id']}"
    values = dict(person["values"])
    with st.expander("📝 Call your own plays: change this person's values and watch every push update"):
        if st.button("Reset to the real example person", key=f"{wkey}_reset"):
            for k in list(st.session_state):
                if k.startswith(wkey + "_f_"):
                    del st.session_state[k]
        cols = st.columns(3)
        for i, f in enumerate(numeric):
            s = b["feature_stats"][f["name"]]
            values[f["name"]] = cols[i % 3].number_input(
                f"{f['label']} ({f['units']})", min_value=float(s["min"]), max_value=float(s["max"]),
                value=float(person["values"][f["name"]]), key=f"{wkey}_f_{f['name']}")
        for j, g in enumerate(groups):
            levels = [lv["label"] for lv in g["levels"]]
            current = next((lv["label"] for lv in g["levels"]
                            if lv["column"] and person["values"][lv["column"]] == 1), g["reference"])
            choice = cols[(len(numeric) + j) % 3].selectbox(
                g["label"], levels, index=levels.index(current), key=f"{wkey}_f_{g['name']}")
            values.update(m10.category_inputs(b, g["name"], choice))
    edited = any(abs(values[k] - person["values"][k]) > 1e-9 for k in values)
    if edited:
        st.warning(f"You've called your own plays, so this is a **made-up person** based on "
                   f"{person['label']}. Same model, same math; the measured value no longer applies.")

    ex = m10.linear_explain(b, values)
    pushes = ex["grouped_pushes"]
    order = sorted(pushes, key=lambda k: -abs(pushes[k]))

    # --- pre-game: starting point ---
    phase("🏟️", "Pre-game", "Where every drive starts")
    c1, c2 = st.columns([1, 2])
    c1.metric("Starting point", fmt(ex["baseline"], u))
    c2.markdown(
        f"Every prediction kicks off from the same spot: **{fmt(ex['baseline'], u)}**, the model's "
        f"prediction for a person with the **training average** of every feature (averaged over the "
        f"{b['n_train']:,} people it learned from).\n\n"
        "**Why not the intercept?** The intercept is the prediction for someone with every feature at 0 "
        "(age 0, BMI 0…). That's like starting the drive in the parking lot: nobody plays from there.")
    gator_tip("Think of the starting point as the average starting field position. Every drive (every "
              "person) begins there, then the plays move the ball.")

    # --- 1st quarter: the drive ---
    phase("🏈", "1st Quarter", "Each feature is a play that gains or loses ground")
    st.markdown(f"This **drive chart** starts at the starting point. Each **orange** bar is a play that pushes "
                f"the prediction **up**; each **blue** bar pushes it **down**. Biggest plays first. "
                f"Everything is in {u}.")
    names = ["Starting point"] + [short_label(b, k) for k in order] + ["Prediction"]
    fig = go.Figure(go.Waterfall(
        orientation="v", x=names,
        measure=["absolute"] + ["relative"] * len(order) + ["total"],
        y=[ex["baseline"]] + [pushes[k] for k in order] + [0],
        text=[fmt(ex["baseline"], "")] + [signed(pushes[k]) for k in order] + [fmt(ex["prediction"], "")],
        textposition="outside",
        increasing=dict(marker=dict(color=ORANGE)), decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color=NAVY)), connector=dict(line=dict(color=LIGHT, dash="dot"))))
    path = np.cumsum([ex["baseline"]] + [pushes[k] for k in order])
    lo, hi = min(path.min(), ex["prediction"]), max(path.max(), ex["prediction"])
    if not edited:
        lo, hi = min(lo, person["observed"]), max(hi, person["observed"])
    pad = max((hi - lo) * 0.25, abs(ex["baseline"]) * 0.02)
    fig.update_yaxes(title=f"Prediction ({u})", range=[lo - pad, hi + pad])
    fig.update_layout(title="Drive chart: starting point → plays → prediction")
    if not edited:
        fig.add_hline(y=person["observed"], line_dash="dash", line_color=GREEN,
                      annotation_text=f"actually measured: {fmt(person['observed'], u)}",
                      annotation_position="top left")
    chart(fig, 430)
    terms = " ".join(f"{'+' if pushes[k] >= 0 else '−'} {abs(pushes[k]):,.1f}" for k in order)
    st.markdown(f"<div class='scoreboard'>SCOREBOARD &nbsp; {ex['baseline']:,.1f} {terms} = "
                f"<b>{fmt(ex['prediction'], u)}</b></div>", unsafe_allow_html=True)
    up_k, down_k = max(pushes, key=pushes.get), min(pushes, key=pushes.get)
    lines = []
    if pushes[up_k] > 0:
        lines.append(f"⬆️ **Biggest push up:** {short_label(b, up_k)} ({signed(pushes[up_k], u)}), "
                     "the tallest orange bar.")
    if pushes[down_k] < 0:
        lines.append(f"⬇️ **Biggest push down:** {short_label(b, down_k)} ({signed(pushes[down_k], u)}), "
                     "the longest blue bar.")
    st.markdown("  \n".join(lines))
    refs = "; ".join(f"**{g['label']}** is one menu with one push (the model's reference group: "
                     f"{g['reference']})" for g in groups)
    st.caption(f"{refs}. Pushes are measured from the training average, which mixes every group.")
    if not edited:
        st.caption(f"Final vs measured: the model was off by {signed(person['observed'] - ex['prediction'], u)} "
                   f"for this person. Its typical miss on new people (validation RMSE) is "
                   f"{fmt(lin['rmse']['validation'], u)}.")

    # --- 2nd quarter: playbook vs this play ---
    phase("📋", "2nd Quarter", "The playbook rule vs. the yards on this play")
    st.markdown(
        "A **coefficient** is a playbook rule the model learned: how much the prediction changes for "
        "each 1 unit of a feature, holding the others fixed. **It's the same for everyone.**\n\n"
        "A **push** is what that rule gains for *this* person:\n\n"
        "**push = coefficient × (this person's value − the training average)**\n\n"
        "A person right at the average gets a push of 0, no matter how big the coefficient is.")
    pick = st.selectbox("Pick a play to break down", order, format_func=lambda k: label_of(b, k),
                        key=f"{wkey}_unpack")
    gnames = {g["name"]: g for g in groups}
    if pick in gnames:
        g = gnames[pick]
        rows = []
        for lv in g["levels"]:
            if not lv["column"]:
                continue
            c, ref, v = lin["coefficients"][lv["column"]], lin["reference"][lv["column"]], values[lv["column"]]
            rows.append({f"Group (compared with {g['reference']})": lv["label"],
                         f"Coefficient ({u})": f"{c:+,.2f}", "This person (1 = yes)": f"{v:.0f}",
                         "Share in training": f"{ref:.0%}", f"Push ({u})": f"{c * (v - ref):+,.2f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            f"{g['label']} is one menu, so its pieces combine into **one push: {signed(pushes[pick], u)}**. "
            f"Each coefficient compares a group with the **reference group ({g['reference']})**, holding the "
            "other features fixed. Pushes, though, are measured from the training *average*, which mixes "
            f"every group, so even someone in the {g['reference']} group gets a small push.")
    else:
        c, ref, v = lin["coefficients"][pick], lin["reference"][pick], values[pick]
        fu = feat_units(b, pick)
        a, bb, cc = st.columns(3)
        a.metric("Playbook rule (coefficient)", f"{c:+,.2f} {u} per {fu}")
        bb.metric("This person vs average", f"{v - ref:+,.1f} {fu}", help=f"{v:,.1f} vs {ref:,.1f} {fu}")
        cc.metric("Yards on this play (push)", signed(pushes[pick], u))
        st.markdown(f"<div class='scoreboard'>{c:+,.2f} × ({v:,.1f} − {ref:,.1f}) = "
                    f"<b>{signed(pushes[pick], u)}</b></div>", unsafe_allow_html=True)
        st.markdown("**Same rule, different yards.** The coefficient never changes; each person's push does:")
        rows = [{"Person": f"{p['label']} ({p['description']})",
                 f"Value ({fu})": f"{p['values'][pick]:,.1f}",
                 f"Coefficient ({u} per {fu})": f"{c:+,.2f}",
                 f"Push ({u})": f"{c * (p['values'][pick] - ref):+,.1f}"} for p in b["people"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    gator_tip("The playbook says what a play is worth per yard. Whether it gains a lot today depends on the "
              "field position: a person far from average gets a big push; one near average barely moves.")

    # --- halftime: player of the game vs season MVP ---
    phase("⏸️", "Halftime film review", "Player of the game ≠ season MVP")
    top_up = max(pushes, key=pushes.get)
    te = lin["typical_effect_per_sd"]
    st.markdown(
        f"For this person, the biggest upward push is **{short_label(b, top_up)} ({signed(pushes[top_up], u)})**. "
        "That's the *player of the game*: it depends on how unusual this one person is.\n\n"
        "To judge features *over the whole season*, use the **typical effect**: coefficient × one standard "
        "deviation of the feature. Raw coefficients can't be compared directly, because features are in "
        "different units (years vs cm vs kg/m²).")
    te_order = sorted(te, key=lambda k: abs(te[k]))
    fig2 = go.Figure(go.Bar(
        x=[te[k] for k in te_order], y=[short_label(b, k) for k in te_order], orientation="h",
        marker_color=[ORANGE if te[k] > 0 else BLUE for k in te_order],
        text=[signed(te[k], u) for k in te_order], textposition="auto"))
    fig2.update_layout(title="Season stats: typical effect of a 1-standard-deviation difference")
    fig2.update_xaxes(title=f"Change in prediction ({u})")
    chart(fig2, 280)

    # --- postgame ---
    phase("📰", "Post-game", "What the game plan can't do")
    if b["id"] == "fev1":
        st.warning("**This playbook only runs straight lines.** Lungs grow through childhood and shrink later "
                   "in life. One straight line can't bend to follow both, so for children (Example person 1 "
                   "is a 9-year-old) the prediction can be far too high. That's the model's limit, not a bug.")
    else:
        st.markdown("A linear model only runs straight-line plays: each feature always adds the same amount "
                    "per unit. Switch to **Lung function** to see a case where that breaks down.")
    st.caption("Pushes describe patterns (associations) in survey data, not causes. Say \"people with higher X "
               "tend to get higher predictions\", not \"raising X would raise it\".")


# =============================================================================
# 2. WISDOM OF THE FOREST (random forest) · THE SCOUTING COMMITTEE
# =============================================================================
def forest_page() -> None:
    b, person = pick_person()
    u, rf = b["units"], b["forest"]
    N = rf["n_trees"]
    pf = rf["people"][person["id"]]
    trees = np.array(pf["tree_predictions"])

    kickoff_header("Wisdom of the Forest: The Scouting Committee",
                   "Random forest: how many decision trees combine into one prediction",
                   f"The forest asks {N} slightly different trees (scouts) for a guess, then averages "
                   "their guesses.", b, person)

    # --- training camp: why scouts differ ---
    phase("🏋️", "Training camp", "Why the scouts don't think alike")
    st.markdown("Each scout is a **decision tree**: a checklist of yes/no questions about a person "
                "(\"Is age ≤ 46.5?\") that ends with a guess. Two things make every tree's checklist different:")
    share = np.mean(rf["bootstrap_unique_training_rows"]) / rf["n_training_rows"]
    a, c = st.columns(2)
    a.markdown(
        f"**1. Each scout studied different film.** Each tree learns from its own random sample: "
        f"{rf['n_training_rows']:,} draws *with replacement* from the training people (a **bootstrap "
        f"sample**). Some people get picked twice, others not at all, so a tree sees only about "
        f"**{share:.0%}** of the distinct people.")
    c.markdown(
        f"**2. Each scout tracks only a few stats at a time.** At every yes/no question, a tree may choose "
        f"from just **{rf['features_tried_per_split']} of the {rf['n_features']} features**, picked at random. "
        "So the trees don't even agree on their first question:")
    feats = pd.Series([r["feature"] for r in rf["root_split"]]).map(lambda k: short_label(b, k)).value_counts()
    fig = go.Figure(go.Bar(x=feats.values, y=feats.index, orientation="h", marker_color=BLUE,
                           text=feats.values, textposition="auto"))
    fig.update_layout(title=f"The first question each of the {N} trees asks")
    fig.update_xaxes(title="Number of trees")
    chart(fig, 240)

    # --- pre-game: meet the scouts ---
    phase("🎙️", "Pre-game", "Each scout makes a call")
    st.markdown("A tree follows its questions until it lands on a **leaf**. Its guess is the average outcome "
                "of the training people who landed on that same leaf. Here are the first five scouts' calls "
                f"for this person (in {u}):")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        rs = rf["root_split"][i]
        col.metric(f"Tree {i + 1} ({u})", f"{trees[i]:,.1f}")
        col.caption(f"Opens with: {short_label(b, rs['feature'])} ≤ {rs['threshold']:g}?")
    with st.expander("Follow Tree 1's checklist for this person, question by question"):
        t1 = pf["traced_trees"][0]
        for n, s in enumerate(t1["steps"], 1):
            st.markdown(f"{n}. Is **{short_label(b, s['feature'])}** ≤ {s['threshold']:g}? → this person is "
                        f"**{s['goes']} {s['threshold']:g}**")
        st.markdown(f"🍃 Leaf reached. Tree 1 calls **{fmt(t1['leaf_value'], u)}**.")

    # --- game time: average the calls ---
    phase("🏈", "Game time", "Add scouts one at a time and average their calls")
    n = st.slider("Scouts (trees) in the average, always in the same fixed order", 1, N, 5,
                  key=f"rf_n_{b['id']}_{person['id']}")
    partial = float(np.mean(trees[:n]))
    if n < N:
        st.markdown(f"<div class='scoreboard'>PARTIAL FOREST &nbsp; average of the first {n} of {N} trees = "
                    f"<b>{fmt(partial, u)}</b><br>FULL FOREST &nbsp;&nbsp;&nbsp; average of all {N} trees = "
                    f"<b>{fmt(pf['prediction'], u)}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='scoreboard'>FULL FOREST &nbsp; average of all {N} trees = "
                    f"<b>{fmt(pf['prediction'], u)}</b> &nbsp;← the random forest's prediction</div>",
                    unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        jitter = np.random.default_rng(0).uniform(-0.35, 0.35, N)
        inc = np.arange(N) < n
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trees[~inc], y=jitter[~inc], mode="markers", name="not in the average yet",
                                 marker=dict(color=LIGHT, size=8)))
        fig.add_trace(go.Scatter(x=trees[inc], y=jitter[inc], mode="markers", name="in the average",
                                 marker=dict(color=BLUE, size=9),
                                 text=[f"Tree {i + 1}" for i in np.where(inc)[0]],
                                 hovertemplate="%{text}: %{x:.1f} " + u + "<extra></extra>"))
        fig.add_vline(x=partial, line_color=ORANGE, line_width=3,
                      annotation_text=("partial forest" if n < N else "forest") + f": {partial:,.1f}",
                      annotation_position="top")
        fig.add_vline(x=person["observed"], line_color=GREEN, line_dash="dash",
                      annotation_text=f"measured: {person['observed']:,.1f}", annotation_position="bottom")
        fig.update_yaxes(visible=False, range=[-1, 1])
        fig.update_xaxes(title=f"Tree's guess ({u})")
        fig.update_layout(title="Every dot is one scout's call")
        chart(fig, 330)
    with right:
        run = np.cumsum(trees) / np.arange(1, N + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, N + 1), y=trees, mode="markers", name="one tree's guess",
                                 marker=dict(color=LIGHT, size=5)))
        fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=run[:n], mode="lines", name="running average (partial forest until all 100)",
                                 line=dict(color=ORANGE, width=3)))
        fig.add_hline(y=person["observed"], line_color=GREEN, line_dash="dash")
        fig.update_xaxes(title="Number of trees averaged", range=[0, N + 1])
        fig.update_yaxes(title=f"Prediction ({u})")
        fig.update_layout(title="One scout jumps around; the committee settles down")
        chart(fig, 330)
    gator_tip("One scout can overrate or underrate a player because of the film they happened to watch. "
              "Different scouts make different mistakes, so averaging many calls cancels much of the noise. "
              "Steadier doesn't mean right, though: check the measured line.")

    # --- post-game review: disagreement vs uncertainty ---
    phase("📰", "Post-game review", "Scouts arguing ≠ a confidence interval")
    lo, hi = np.percentile(trees, [10, 90])
    st.markdown(
        f"For this person, the scouts' calls run from **{fmt(trees.min(), u)} to {fmt(trees.max(), u)}** "
        f"(middle 80% of trees: {lo:,.1f} to {hi:,.1f} {u}). That spread is **disagreement between trees**. "
        "It is **not** a confidence or prediction interval: the scouts studied overlapping film of the same "
        "people, so they share blind spots. Their spread isn't a calibrated measure of how far off the "
        "forest might be.")
    rows = []
    for p in b["people"]:
        tp = np.array(rf["people"][p["id"]]["tree_predictions"])
        rows.append({"Person": f"{p['label']} ({p['description']})",
                     f"Forest prediction ({u})": f"{rf['people'][p['id']]['prediction']:,.1f}",
                     f"Disagreement: lowest–highest tree ({u})": f"{tp.min():,.1f} – {tp.max():,.1f}",
                     f"Measured ({u})": f"{p['observed']:,.1f}", "Why chosen": p["why_chosen"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        f"A fairer guide to real error is the **season record against new opponents**: on the validation "
        f"people, the forest's typical miss (RMSE) is **{fmt(rf['rmse']['validation'], u)}**. Compare Example "
        "person 4 (scouts disagree most) with Example person 5 (scouts agree most): agreement makes the "
        "forest *steady*, not necessarily *right*.")
    with st.expander("Bonus film: does a bigger committee help against new opponents?"):
        vr = rf["validation_rmse_by_trees"]
        fig = go.Figure(go.Scatter(x=np.arange(1, len(vr) + 1), y=vr, mode="lines", line=dict(color=BLUE, width=3)))
        fig.update_xaxes(title="Number of trees in the (partial) forest")
        fig.update_yaxes(title=f"Validation RMSE ({u})")
        fig.update_layout(title="Typical miss on validation people as trees are added")
        chart(fig, 300)
        st.caption(f"One tree misses by {fmt(vr[0], u)} on average; all {N} trees: {fmt(vr[-1], u)}.")
    tr = rf["training_target_range"]
    st.caption(f"A forest can't call a result it has never seen: every leaf is an average of training outcomes, "
               f"so predictions stay between {fmt(tr[0], u)} and {fmt(tr[1], u)}.")


# =============================================================================
# 3. ONE CORRECTION AT A TIME (gradient boosting) · THE SEASON
# =============================================================================
def boosting_page() -> None:
    b, person = pick_person()
    u, gb = b["units"], b["boosting"]
    p = gb["people"][person["id"]]
    staged, corr = p["staged"], p["corrections"]
    R = len(corr)
    tr, va = gb["rmse"]["train"], gb["rmse"]["validation"]
    best = gb["best_validation_round"]

    kickoff_header("One Correction at a Time: The Season",
                   "Gradient boosting: building a prediction round by round, and knowing when to stop",
                   "Boosting starts at the average. Each small tree fixes part of the remaining error, until "
                   "more trees stop helping on new people.", b, person)

    # --- kickoff ---
    phase("🏋️", "Training camp", "Round 0 is the average")
    st.markdown(f"Before any trees, the best single guess for everyone is the average outcome of the training "
                f"people: **{fmt(gb['init'], u)}**. That's round 0.")

    # --- practice reps ---
    phase("🔁", "Practice reps", "Each round, a small tree fixes part of what's still wrong")
    st.markdown(
        "After each round, the coaches review what the model still gets wrong on the **training** people: "
        "the **residuals** (measured − predicted). The next small tree (3 questions deep) is trained to "
        f"predict those residuals. Only **{gb['learning_rate']:.0%}** of its correction is used (the "
        "**learning rate**): small adjustments, rep after rep, instead of one big overhaul.")
    rkey = f"gb_r_{b['id']}_{person['id']}"
    if rkey not in st.session_state:
        st.session_state[rkey] = 0

    def move(d: int) -> None:
        st.session_state[rkey] = int(min(R, max(0, st.session_state[rkey] + d)))

    bc = st.columns([1, 1, 1, 5])
    bc[0].button("◀ Back", key=rkey + "_b", on_click=move, args=(-1,))
    bc[1].button("Next rep ▶", key=rkey + "_n", on_click=move, args=(1,))
    bc[2].button("Reset", key=rkey + "_0", on_click=move, args=(-R,))
    r = bc[3].slider("Round", 0, R, key=rkey)
    if r == 0:
        st.markdown(f"<div class='scoreboard'>ROUND 0 &nbsp; starting point = <b>{fmt(gb['init'], u)}</b>"
                    "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='scoreboard'>ROUND {r} &nbsp; {staged[r - 1]:,.2f} (after round {r - 1}) "
            f"{'+' if corr[r - 1] >= 0 else '−'} {abs(corr[r - 1]):,.2f} (tree {r}'s correction) = "
            f"<b>{fmt(staged[r], u, 2)}</b> &nbsp;(≈ {fmt(staged[r], u)})</div>", unsafe_allow_html=True)
        st.caption(f"The correction is used exactly as shown: it already includes the learning rate. "
                   f"Measured value: {fmt(person['observed'], u)} (gap now {signed(person['observed'] - staged[r], u)}).")
    span = st.radio("Show", ["First 50 rounds", f"All {R} rounds"], horizontal=True, key="gb_span")
    xmax = 50 if span.startswith("First") else R
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(xmax + 1), y=staged[:xmax + 1], mode="lines",
                             name="prediction for this person", line=dict(color=BLUE, width=3)))
    fig.add_hline(y=person["observed"], line_color=GREEN, line_dash="dash",
                  annotation_text=f"measured: {person['observed']:,.1f}", annotation_position="top left")
    if r <= xmax:
        fig.add_trace(go.Scatter(x=[r], y=[staged[r]], mode="markers", name=f"round {r}",
                                 marker=dict(color=ORANGE, size=13)))
    fig.update_xaxes(title="Round (number of trees added)")
    fig.update_yaxes(title=f"Prediction ({u})")
    fig.update_layout(title="This person's prediction, rep by rep")
    chart(fig, 340)
    gator_tip("Like reviewing film after every practice rep: the coaches don't redraw the whole playbook, "
              "they fix part of what's still going wrong. Not every adjustment helps every player: each tree "
              "fixes the training residuals overall, not one person's.")

    # --- game day vs practice ---
    phase("🤝", "Scrimmage", "Practice scores vs. scrimmage scores")
    st.markdown(
        f"- **Training error** = how the model does in practice, on the people the trees learned from.\n"
        f"- **Validation error** = how it does in a scrimmage against another team: **{b['n_validation']:,} "
        "people held back** and never used to fit any tree.\n\n"
        "Both are measured as **RMSE** (root mean squared error): the typical size of a miss, in "
        f"{u}. Practice error keeps dropping every round. Scrimmage error drops at first, then stops improving.")

    # --- two-minute drill / overtime ---
    phase("⏱️", "Two-minute drill", "Pick the round to stop")
    stop = st.slider("Your stopping round", 1, R, R, key=f"gb_stop_{b['id']}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(R + 1), y=tr, name="training RMSE (practice)",
                             line=dict(color=GRAY, width=3)))
    fig.add_trace(go.Scatter(x=np.arange(R + 1), y=va, name="validation RMSE (scrimmage)",
                             line=dict(color=BLUE, width=3)))
    fig.add_vline(x=best, line_color=GREEN, line_dash="dash",
                  annotation_text=f"lowest validation error: round {best}", annotation_position="top right")
    fig.add_trace(go.Scatter(x=[stop], y=[va[stop]], mode="markers", name="your stopping round",
                             marker=dict(color=ORANGE, size=14, symbol="diamond")))
    fig.update_xaxes(title="Round (number of trees)")
    fig.update_yaxes(title=f"RMSE, typical miss ({u})", rangemode="tozero")
    fig.update_layout(title="Practice vs scrimmage: error by round")
    chart(fig, 380)
    a, c, d = st.columns(3)
    a.metric(f"Scrimmage (validation) error at round {stop}", fmt(va[stop], u))
    c.metric(f"Best scrimmage error (round {best})", fmt(gb["best_validation_rmse"], u))
    d.metric(f"Practice (training) error at round {stop}", fmt(tr[stop], u))
    gap = va[stop] - gb["best_validation_rmse"]
    if stop == best:
        st.success(f"Clock management on point: round {best} is where validation error is lowest. "
                   "That's the validation-based choice.")
    elif stop > best:
        st.warning(f"🕐 **Overtime that doesn't help.** You're {stop - best} rounds past the validation "
                   f"minimum. Practice error keeps falling, but scrimmage error is {gap:,.2f} {u} worse than "
                   "its best: the extra trees mostly learned noise, quirks of the training people that new "
                   "people don't share. That's **overfitting**.")
    else:
        st.info(f"You took a knee {best - stop} rounds early. Scrimmage error could still drop by "
                f"{gap:,.2f} {u} with more rounds.")
    st.markdown(f"**Why not stop where practice error is lowest?** That's always the final round ({R}), because "
                "the trees keep memorizing the practice squad. Scrimmage (validation) error tells you when the "
                "corrections stop helping on *new* people.")

    # --- bowl game: validation vs test ---
    phase("🏆", "Game day", "The scrimmage is not the real game")
    st.markdown(f"We used the scrimmage (validation) results to *choose* round {best}, so it's a "
                "**validation-based choice**, not a test result. Like a depth chart set from scrimmage film, "
                "it's useful but a bit flattering. The real game, a true test, needs opponents that were never "
                "used for any decision. This app doesn't have one, so it never reports a test score.")
    with st.expander("Bonus film: smaller adjustments (learning rate 0.02)"):
        cmp = gb["comparison_learning_rate_0_02"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(R + 1), y=va, name="learning rate 0.1",
                                 line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=np.arange(len(cmp["rmse"]["validation"])), y=cmp["rmse"]["validation"],
                                 name="learning rate 0.02 (separate model)", line=dict(color=ORANGE, width=3)))
        fig.update_xaxes(title="Round")
        fig.update_yaxes(title=f"Validation RMSE ({u})")
        chart(fig, 300)
        st.caption(f"A separately trained model taking smaller steps reaches its lowest validation error at round "
                   f"{cmp['best_validation_round']} ({fmt(cmp['best_validation_rmse'], u)}), versus round {best} "
                   f"({fmt(gb['best_validation_rmse'], u)}) at 0.1. Smaller adjustments need more rounds.")


# =============================================================================
# KICKOFF (home page)
# =============================================================================
def home_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🐊 Explain the Model: Gator Football Edition")
    st.markdown(
        "Three ways a computer can predict a health number, taught through a football season. "
        "You already know a model **learns from data to predict a number** and should be **checked on new "
        "data**. Here you'll see *how* each model turns one person's data into a prediction.")
    st.markdown(
        "<div class='field'>The data: real people from the CDC's NHANES health survey. Pick a dataset and one "
        "of five example people in the sidebar of any section.</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📋 The Game Plan\n**Linear regression.** Start from the average, then each feature "
                    "is a play that pushes the prediction up or down.")
        st.page_link(PAGES["linear"], label="Go to Push & Pull →", icon="🏈")
    with c2:
        st.markdown("#### 🎙️ The Scouting Committee\n**Random forest.** 100 scouts who studied different film "
                    "each make a call; the forest averages them.")
        st.page_link(PAGES["forest"], label="Go to Wisdom of the Forest →", icon="🌳")
    with c3:
        st.markdown("#### ⏱️ The Season\n**Gradient boosting.** Start at the average, fix a little more each "
                    "practice rep, and stop when scrimmage results stop improving.")
        st.page_link(PAGES["boosting"], label="Go to One Correction at a Time →", icon="🔁")
    st.caption("The football story is an analogy only; every number shown is real model output. "
               "Unofficial fan-themed class project, not affiliated with the University of Florida.")


st.set_page_config(page_title="Explain the Model · Gator Edition", page_icon="🐊", layout="wide")
PAGES = {
    "linear": st.Page(linear_page, title="Push & Pull: The Game Plan", url_path="linear"),
    "forest": st.Page(forest_page, title="Wisdom of the Forest: The Scouting Committee", url_path="forest"),
    "boosting": st.Page(boosting_page, title="One Correction at a Time: The Season", url_path="boosting"),
}
st.navigation([st.Page(home_page, title="Kickoff", icon="🐊", default=True)] + list(PAGES.values())).run()
