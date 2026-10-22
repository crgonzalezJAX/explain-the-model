# Explain the Model: Gator Football Edition 🐊

**App:** _paste your public app link here_

| Section | Direct link | My viewer should walk away knowing… |
|---|---|---|
| Push & Pull: The Game Plan (linear regression) | _app link_/linear | The model starts from an average prediction and adds one push per feature (coefficient × how far this person is from average); starting point + all pushes = prediction. |
| Wisdom of the Forest: The Scouting Committee (random forest) | _app link_/forest | The forest averages 100 trees that each learned from different samples and features; their spread is disagreement, not a confidence interval. |
| One Correction at a Time: The Season (gradient boosting) | _app link_/boosting | Boosting starts at the average and each small tree fixes part of the remaining training error; stop at the round where validation error is lowest. |

An app that explains three regression models to an anonymous visitor who knows about machine learning and regression, but not how these algorithms work. Datasets: NHANES fasting glucose, blood pressure and lung function. Each model is taught through a phase of a football season (game plan, scouting, practice reps, game day, overtime). Unofficial fan theme, not affiliated with the University of Florida.

Submit the link to **this repository** on Canvas. Reviewers open the app link above.

## What's here
| File | What it is |
|---|---|
| `AGENT_BRIEF.md` | Rules for your AI coding agent (have it read this first) |
| `SETUP.md` | How to run and publish (GitHub Pages or Streamlit) |
| `check.md` | Numbers your finished app must still show |
| `index.html`, `app.js`, `m10.js` | GitHub Pages route |
| `streamlit_app.py`, `m10.py`, `requirements.txt` | Streamlit route |
| `outputs/` | Reference model outputs (field guide: `outputs/SCHEMA.md`) |
| `data/` | The three datasets and the data dictionary |

## Run locally
- GitHub Pages route: `python -m http.server 8000`, then open <http://localhost:8000>
- Streamlit route: `uv run --with streamlit,pandas,numpy streamlit run streamlit_app.py`
