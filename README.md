# IMDb Review Analysis with Phi-3

This project uses Phi-3 with Ollama to analyze IMDb reviews. The model predicts
if a review is positive or negative. It also finds movie titles and real people
mentioned in the review.

I used 100 reviews from the IMDb test split: 50 positive and 50 negative.
I tested the same reviews with two methods:

- **Prompt-only:** The prompt asks the model to return JSON.
- **Schema-constrained:** The same prompt is used, and Ollama also receives a
  JSON schema through the `format` parameter.

## Setup

Install Python and Ollama. Open PowerShell in the project folder and run:

```powershell
ollama pull phi3:latest
py -m pip install -r requirements.txt
```

Keep Ollama running while you use the script.

## Run the Project

```powershell
py .\imdb_project.py prepare
py .\imdb_project.py run
py .\imdb_project.py score
```

`prepare` selects the 100 reviews. `run` tests both methods and saves the
predictions. Before running `score`, make sure `gold_movies` and `gold_people`
are filled in `outputs_nlp1/entity_review.csv`. Use `[]` when there are no names
of that type. `score` uses the saved predictions to calculate entity F1.

To test the model on five short examples at two temperatures, run:

```powershell
py .\imdb_project.py demo --temperature 0
py .\imdb_project.py demo --temperature 1.0
```

## Results

| Metric | Prompt-only | Schema-constrained |
|---|---:|---:|
| Raw JSON parsing success | 85% | 100% |
| JSON parsing after cleaning | 97% | 100% |
| Sentiment accuracy | 89% | 92% |

`outputs_nlp1/results.csv` contains the model predictions.
`outputs_nlp1/metrics.json` contains the evaluation scores.
See [the technical report](report.pdf) for the method, chart, and discussion.