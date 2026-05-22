# Competitive Intelligence Agent — Flask UI

A lightweight Flask web app that calls a **multi-agent competitive intelligence pipeline** deployed on **Vertex AI Agent Engine**.

The agents (Market Scanner → Sentiment Analyzer → Pricing Intelligence → Report Generator) are built and deployed via the Jupyter notebook. This app is the UI layer that calls the deployed agent.

---

## Architecture

```
Browser → Flask UI → Vertex AI Agent Engine → Gemini 2.5 Pro
```

---

## Project Structure

```
├── app/
│   ├── server.py          # Flask server — calls deployed agent
│   ├── utils.py           # Input sanitization and query helpers
│   └── templates/
│       └── index.html     # Web UI
├── notebooks/
│   └── competitive_intelligence.ipynb   # Build & deploy agents here
├── .env.example           # Environment variable template
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- A GCP project with **Vertex AI API** enabled
- The agent already deployed via the notebook (you need the resource name)

---

## Running in GitHub Codespaces

### Step 1 — Install Google Cloud SDK

`gcloud` is not pre-installed in Codespaces. Install it:

```bash
curl https://sdk.cloud.google.com | bash
```

Restart the shell after install:

```bash
exec -l $SHELL
```

Verify:

```bash
gcloud --version
```

### Step 2 — Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

This opens a URL — open it in your browser, sign in with your Google account, and paste the verification code back into the terminal.

### Step 3 — Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
AGENT_ENGINE_RESOURCE_NAME=projects/your-project/locations/us-central1/reasoningEngines/1234567890
```

> Get `AGENT_ENGINE_RESOURCE_NAME` from the notebook Section 10 output after deploying the agent.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the app

```bash
python -m flask --app app.server run --port 5000
```

Open the forwarded port in your browser and enter a competitor name to generate a report.

---

## Running Locally

Same steps as above, but skip the gcloud SDK install if you already have it.

```bash
gcloud auth application-default login
cp .env.example .env   # fill in your values
pip install -r requirements.txt
python -m flask --app app.server run --port 5000
```

---

## Deploying the Agents (Notebook)

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/03sarath/A2A/blob/master/notebooks/competitive_intelligence.ipynb)

Run all sections top to bottom. Section 10 deploys the agent to Vertex AI Agent Engine and prints the resource name — copy that into your `.env`.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Yes | Your GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Yes | Region, e.g. `us-central1` |
| `AGENT_ENGINE_RESOURCE_NAME` | Yes | Full resource name of deployed agent |
| `STAGING_BUCKET` | Notebook only | GCS bucket for deployment staging |
| `PORT` | No | Flask port, defaults to `5000` |
