# Notebook Code Explained — Competitive Intelligence Platform

A beginner-friendly, line-by-line walkthrough of `notebooks/competitive_intelligence.ipynb`.

---

## What Problem Does This Solve?

Tracking competitors manually is slow and expensive. A business analyst has to:
- Read dozens of news articles
- Scan review sites and social media
- Find pricing pages across multiple websites
- Summarize everything into a report for leadership

This notebook automates all of that using **4 AI agents** that work together like a team — each agent does one job, passes its findings to the next, and the final agent writes a complete executive report. The whole pipeline runs in under 2 minutes.

**Real-world use case:** A product manager types "Analyze competitor: OpenAI" and gets back a structured report covering recent news, brand sentiment, pricing tiers, strategic threats, and recommended actions — all pulled live from the web.

---

## Key Concepts to Know Before You Start

### What is an Agent?
An agent is an AI model (like Gemini) that has been given:
1. **An instruction** — what role it plays and what to do
2. **Tools** — capabilities like searching the web

Unlike a simple chatbot, an agent can **take actions** (run a Google search) and **reason** about the results before responding.

### What is Google ADK?
ADK (Agent Development Kit) is Google's open-source framework for building AI agents. It handles:
- Running agents
- Managing conversation memory (sessions)
- Wiring multiple agents together

### What is Vertex AI?
Google Cloud's managed AI platform. We use it to:
- Run Gemini models (the brain of each agent)
- Deploy our agents to the cloud so they run 24/7

### What is a SequentialAgent?
A special type of agent that runs other agents **one after another**, passing the output of each as input to the next. Think of it as a pipeline or assembly line.

### What is async/await?
Python normally runs code line by line. `async/await` lets code run while waiting for a slow operation (like a web search) to finish, instead of freezing and waiting. Agents use this because AI responses take time.

---

## Section 1 — Install Dependencies

```python
%pip install --upgrade -q google-adk==1.9.0 google-genai google-cloud-aiplatform[agent_engines,adk] python-dotenv nest_asyncio
```

**What this does:** Installs all the Python libraries needed. Run this once at the start.

| Package | Why we need it |
|---|---|
| `google-adk==1.9.0` | The Agent Development Kit — the core framework for building agents |
| `google-genai` | Google's Python client for Gemini AI models |
| `google-cloud-aiplatform[agent_engines,adk]` | Vertex AI SDK — needed to deploy agents to the cloud |
| `python-dotenv` | Reads environment variables from a `.env` file |
| `nest_asyncio` | Allows `async` code to run inside Jupyter notebooks (explained in Section 3) |

**The `%` prefix:** This is a Jupyter "magic command". `%pip` installs packages directly into the notebook's Python environment. The `-q` flag means "quiet" (less output). `--upgrade` ensures you get the latest version.

**Why pin `google-adk==1.9.0`?** ADK changes frequently. Pinning to a specific version ensures the code works the same way every time.

---

## Section 2 — Configure Google Cloud

```python
import os
import sys
```

**`import os`** — Gives access to operating system features, specifically environment variables. Environment variables are key-value pairs that store configuration outside of your code (like passwords and project IDs).

**`import sys`** — Gives access to the Python interpreter itself. We use it to detect if we're running in Colab.

```python
GOOGLE_CLOUD_PROJECT  = "your-gcp-project-id"
GOOGLE_CLOUD_LOCATION = "us-central1"
```

**`GOOGLE_CLOUD_PROJECT`** — Your unique GCP project ID. Every Google Cloud resource (Vertex AI models, storage, etc.) belongs to a project for billing and access control. Change this to your actual project ID.

**`GOOGLE_CLOUD_LOCATION`** — The geographic region where your AI models run. `us-central1` is in Iowa, USA. Keeping agents and storage in the same region avoids extra latency and data transfer costs.

**Variable naming convention:** ALL_CAPS names are used for constants — values that don't change during the program's lifetime.

```python
os.environ["GOOGLE_GENAI_USE_VERTEXAI"]  = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"]       = GOOGLE_CLOUD_PROJECT
os.environ["GOOGLE_CLOUD_LOCATION"]      = GOOGLE_CLOUD_LOCATION
```

**`os.environ`** — A Python dictionary that holds environment variables. Setting a key here makes it available to all libraries running in this process.

**`GOOGLE_GENAI_USE_VERTEXAI = "TRUE"`** — This is critical. The `google-genai` library can use two backends:
- Google AI Studio (free tier, consumer)
- Vertex AI (enterprise, your GCP project)

Setting this to `"TRUE"` tells every ADK agent to use **Vertex AI** and bill your GCP project. Without this, agents would try to use the free tier and fail.

```python
if "google.colab" in sys.modules:
    from google.colab import auth
    auth.authenticate_user(project_id=GOOGLE_CLOUD_PROJECT)
```

**`sys.modules`** — A dictionary of all currently loaded Python modules. If `google.colab` is in it, we're running inside Google Colab.

**`auth.authenticate_user()`** — Opens a browser popup asking you to sign in with your Google account. This gives Colab permission to call Google Cloud APIs on your behalf. This step only runs in Colab — on a local machine you'd use `gcloud auth application-default login` in the terminal instead.

```python
print("Configuration:")
print(f"  Project  : {GOOGLE_CLOUD_PROJECT}")
```

**f-strings (`f"..."`)** — A Python string formatting feature. `{GOOGLE_CLOUD_PROJECT}` inside an f-string gets replaced with the actual variable value. Useful for readable output.

---

## Section 3 — Imports

```python
import asyncio
import nest_asyncio
```

**`asyncio`** — Python's built-in library for writing asynchronous (non-blocking) code. ADK agents use `async/await` because AI model calls take time — asyncio lets the program do other things while waiting.

**`nest_asyncio`** — Jupyter notebooks already run their own `asyncio` event loop. Normally you can't run another event loop inside an existing one — it causes an error. `nest_asyncio` patches this limitation so async code works in notebooks.

```python
nest_asyncio.apply()
```

**`.apply()`** — Activates the patch. Call this once before running any async code. Without it, `asyncio.run()` would crash in Jupyter.

```python
from google.adk.agents import Agent, SequentialAgent
```

**`Agent`** — The core class for creating a single AI agent. You give it a model, a name, instructions, and tools.

**`SequentialAgent`** — A special agent that runs other agents one after another. This is what orchestrates our 4-agent pipeline.

```python
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
```

These four work together to give the agent a "brain" for local testing:

| Class | What it stores | Real-world analogy |
|---|---|---|
| `InMemorySessionService` | The current conversation history | Short-term memory |
| `InMemoryArtifactService` | Files/data the agent produces | A scratchpad |
| `InMemoryMemoryService` | Long-term facts across sessions | A notebook |
| `Runner` | Connects everything and runs the agent | The engine |

**"InMemory"** means data is stored in RAM — it disappears when the notebook restarts. Perfect for testing. In production (Vertex AI Agent Engine), this is replaced with Google's managed, persistent storage automatically.

```python
from google.adk.tools import google_search
```

**`google_search`** — A pre-built tool from ADK that lets agents search the web using Google Search. When an agent has this in its `tools` list, it can call Google Search as part of answering a question.

```python
from google.genai import types
```

**`types`** — Contains data structures for communicating with Gemini models. We use `types.Content` and `types.Part` to wrap our text messages in the format Gemini expects.

---

## Section 4 — Define the 4 Specialist Agents

```python
MODEL = "gemini-2.5-pro"
```

A constant storing the model name. All 4 agents use the same model. `gemini-2.5-pro` is Google's most capable reasoning model — important for tasks that require reading search results and synthesizing information.

Using a constant means you only change the model name in one place if you ever want to switch.

### Agent 1: Market Scanner

```python
market_scanner = Agent(
    model=MODEL,
    name="market_scanner",
    instruction="""...""",
    tools=[google_search],
)
```

**`model=MODEL`** — Which AI model powers this agent. Gemini 2.5 Pro.

**`name="market_scanner"`** — A unique identifier. ADK uses this name in logs so you can see which agent is currently running. Also used by `SequentialAgent` to pass outputs between agents.

**`instruction="""..."""`** — This is the most important part. The instruction is a system prompt — it tells the AI model what role it plays, what steps to follow, and what format to return. Think of it as a job description for the agent.

Key things in this instruction:
- `"You are a market intelligence specialist"` — sets the persona
- The numbered steps — tell the agent exactly what to search for
- The output format (`**MARKET SCAN**`) — ensures consistent output that the next agent can read

**`tools=[google_search]`** — A list of tools this agent can use. `google_search` lets it call Google Search. The model decides when and what to search for based on the instruction.

**Why separate agents for each task?** Each agent has a narrow focus. This produces better results than one agent trying to do everything at once. It also makes debugging easier — if the pricing data is wrong, you know it's Agent 3.

### Agent 2: Sentiment Analyzer

```python
sentiment_analyzer = Agent(
    model=MODEL,
    name="sentiment_analyzer",
    instruction="""...""",
    tools=[google_search],
)
```

Same structure as Agent 1. Key differences in the instruction:
- Searches for `"reviews"`, `"complaints"`, `"customer feedback"`, `"analyst opinion"` — different search terms targeting sentiment signals
- Returns `**SENTIMENT ANALYSIS**` section with a 1-10 score

**How does it know which competitor to analyze?** The instruction says `"Read the conversation to identify the competitor name"`. Because agents share conversation context in a `SequentialAgent`, this agent can see the user's original message ("Analyze competitor: OpenAI") and Agent 1's output, so it knows the competitor name.

### Agent 3: Pricing Intelligence

```python
pricing_intelligence = Agent(
    model=MODEL,
    name="pricing_intelligence",
    instruction="""...""",
    tools=[google_search],
)
```

Searches for pricing-specific terms. The instruction asks for a table format:
```
Tier | Price | Key Features
```
Structured output makes it easy for Agent 4 (the report writer) to include this data cleanly in the final report.

### Agent 4: Report Generator

```python
report_generator = Agent(
    model=MODEL,
    name="report_generator",
    instruction="""...""",
    tools=[],   # <-- no search tool
)
```

**`tools=[]`** — Empty list. This agent has NO tools. It cannot search the web. This is intentional.

**Why no tools?** By this point, the conversation already contains the Market Scan, Sentiment Analysis, and Pricing Intelligence sections produced by Agents 1-3. Agent 4's job is purely to read and synthesize that existing information into an executive report. Giving it search access would cause it to search again unnecessarily, wasting time and money.

The instruction says `"Do NOT perform any new searches"` to make this explicit.

The instruction also defines an exact output structure:
- Executive Summary
- Strategic Threats
- Opportunities
- Recommended Actions
- Competitive Threat Level (Low / Medium / High)

This guarantees the final report always looks the same regardless of the competitor.

---

## Section 5 — Wire the Host Agent (SequentialAgent)

```python
host_agent = SequentialAgent(
    name="competitive_intel_host",
    description="Orchestrates market scan, sentiment, pricing, and report generation.",
    sub_agents=[
        market_scanner,
        sentiment_analyzer,
        pricing_intelligence,
        report_generator,
    ],
)
```

**`SequentialAgent`** — Runs its `sub_agents` list in order, one by one. Each agent's output is added to the shared conversation context before the next agent starts.

**`name`** — Identifier for the host agent itself.

**`description`** — A human-readable summary of what this agent does. Also used by ADK internally.

**`sub_agents=[...]`** — The ordered list. The order matters — `report_generator` must be last because it depends on the outputs of the first three.

**How context flows:**
```
User message → market_scanner runs → adds MARKET SCAN to context
             → sentiment_analyzer runs → reads context, adds SENTIMENT ANALYSIS
             → pricing_intelligence runs → reads context, adds PRICING INTELLIGENCE
             → report_generator runs → reads all three sections, writes final report
```

```python
print(f"   Pipeline: {' → '.join(a.name for a in host_agent.sub_agents)}")
```

**`' → '.join(...)`** — Joins a list of strings with ` → ` between each. Prints:
```
market_scanner → sentiment_analyzer → pricing_intelligence → report_generator
```

**`a.name for a in host_agent.sub_agents`** — A Python list comprehension. For each agent `a` in the sub_agents list, get its `.name` attribute. Produces: `['market_scanner', 'sentiment_analyzer', 'pricing_intelligence', 'report_generator']`

---

## Section 6 — Create the Runner

```python
session_service  = InMemorySessionService()
artifact_service = InMemoryArtifactService()
memory_service   = InMemoryMemoryService()
```

Creates three in-memory storage objects. They hold data only while the notebook is running.

**`session_service`** — Tracks conversation history. Each "session" is one conversation with the agent. The agent reads the session to know what was said before.

**`artifact_service`** — Stores binary artifacts (files, images) an agent might produce. We don't use this actively in this notebook, but ADK requires it.

**`memory_service`** — Stores long-term memories across different sessions. Also not heavily used here, but required by the Runner.

```python
runner = Runner(
    app_name="competitive_intel",
    agent=host_agent,
    artifact_service=artifact_service,
    session_service=session_service,
    memory_service=memory_service,
)
```

**`Runner`** — The execution engine. It takes a query, feeds it to the agent, manages the conversation loop, and returns events (intermediate steps + final response).

**`app_name="competitive_intel"`** — A namespace for sessions. Sessions are stored under this name, so different apps don't mix their conversations.

**`agent=host_agent`** — The agent to run. We pass the `SequentialAgent` here, not individual agents.

---

## Section 7 — Helper: Run Agent

```python
async def run_agent(query: str, session_id: str = "notebook_session") -> str:
```

**`async def`** — Declares an asynchronous function. Must be awaited or run with `asyncio.run()`.

**`query: str`** — Type hint. `query` must be a string. Type hints don't enforce types at runtime — they're documentation for the developer.

**`session_id: str = "notebook_session"`** — Optional parameter with a default value. If you don't pass a `session_id`, it uses `"notebook_session"` automatically.

**`-> str`** — Return type hint. This function returns a string (the final report).

```python
existing = await session_service.get_session(
    app_name="competitive_intel",
    user_id="notebook_user",
    session_id=session_id,
)
if not existing:
    await session_service.create_session(
        app_name="competitive_intel",
        user_id="notebook_user",
        session_id=session_id,
    )
```

**Why create a session manually?** ADK 1.9.0 changed behavior — it no longer auto-creates a session when you call `run_async`. Without an existing session, it throws a `ValueError: Session not found`. This `get → if None → create` pattern is the fix.

**`await`** — Pauses this function until `get_session` finishes (it talks to the session storage). Other code can run during this wait.

**`user_id="notebook_user"`** — An identifier for who is having this conversation. In production, this would be a real user ID.

**`session_id`** — A unique ID for this specific conversation. If you use the same session_id across multiple calls, the agent remembers the previous conversation.

```python
message = types.Content(role="user", parts=[types.Part(text=query)])
```

**`types.Content`** — Wraps a message in the format Gemini expects.

**`role="user"`** — Tells Gemini this message came from a human (as opposed to `"model"` for AI responses).

**`parts=[types.Part(text=query)]`** — A message can have multiple parts (text, images, etc.). Here we have one text part containing our query.

```python
final_response = ""
current_agent  = ""

async for event in runner.run_async(
    user_id="notebook_user",
    session_id=session_id,
    new_message=message,
):
```

**`async for`** — Iterates over events as they stream in, one by one. The agent pipeline produces many events (tool calls, intermediate outputs, final responses). We process each as it arrives instead of waiting for everything to finish.

**`runner.run_async()`** — Starts the agent pipeline. Returns an async generator that yields events.

```python
    if hasattr(event, "author") and event.author != current_agent:
        current_agent = event.author
        print(f"\n▶ {current_agent} is working...")
```

**`hasattr(event, "author")`** — Checks if the event has an `author` attribute (not all events do). `hasattr` returns `True`/`False`.

**`event.author`** — The name of the agent that produced this event. For example, `"market_scanner"` or `"report_generator"`.

**`!= current_agent`** — Only print when the agent changes. Without this check, you'd print "working..." dozens of times as events stream in from the same agent.

```python
    if event.is_final_response():
        if event.content and event.content.parts:
            final_response = event.content.parts[0].text
```

**`event.is_final_response()`** — Returns `True` only for the last event from the last agent (the complete answer). There are many intermediate events (tool calls, partial outputs) — we only want the final one.

**`event.content.parts[0].text`** — Gets the text from the first part of the response content. `[0]` gets the first item in the list (Python lists are zero-indexed).

```python
return final_response
```

Returns the complete report text after all 4 agents finish.

---

## Section 8 — Test Agent 1 Individually

```python
single_session_service = InMemorySessionService()

single_runner = Runner(
    app_name="test_market_scanner",
    agent=market_scanner,       # <-- only Agent 1, not the full pipeline
    ...
)
```

Creates a separate, isolated runner with only `market_scanner`. This lets you test one agent without running the full 4-agent pipeline. Good practice: always test agents individually before wiring them together.

```python
async def test_single_agent(agent_runner, single_svc, query, session_id="test_session"):
```

A reusable test function. Takes the runner and session service as parameters so it can work with any single agent — not hardcoded to `market_scanner`.

```python
result = asyncio.run(
    test_single_agent(single_runner, single_session_service, "Analyze competitor: Anthropic")
)
```

**`asyncio.run()`** — The standard way to call an `async` function from regular (non-async) code. In Jupyter, this works because we applied `nest_asyncio.apply()` in Section 3.

**`"Analyze competitor: Anthropic"`** — A test query. The market scanner will search for recent news about Anthropic.

---

## Section 9 — Run the Full Pipeline

```python
COMPETITOR = "Anthropic"

query = (
    f"Perform a comprehensive competitive intelligence analysis for: {COMPETITOR}. "
    f"Cover recent market moves, brand sentiment, and pricing strategy."
)
```

**Why wrap the query in a full sentence?** The agent instruction says agents should "read the conversation to identify the competitor name." A detailed query gives the agent more context and produces better-directed searches.

**String concatenation in parentheses:** Python automatically joins string literals that are adjacent inside parentheses. This is a clean way to write long strings across multiple lines without using `\` or `+`.

```python
report = asyncio.run(run_agent(query, session_id=f"session_{COMPETITOR.lower()}"))
```

**`COMPETITOR.lower()`** — Converts `"Anthropic"` to `"anthropic"`. Session IDs should be lowercase and consistent.

**`f"session_{COMPETITOR.lower()}"`** — Produces `"session_anthropic"`. Using the competitor name in the session ID means each competitor gets its own isolated conversation history.

---

## Section 10 — Deploy to Vertex AI Agent Engine

```python
import vertexai
from vertexai.preview import reasoning_engines
```

New imports needed only for deployment. `reasoning_engines` is the Vertex AI module for deploying agents.

```python
STAGING_BUCKET = f"gs://{GOOGLE_CLOUD_PROJECT}-agent-staging"
```

**`gs://`** — The URI prefix for Google Cloud Storage buckets. Like `https://` for websites.

**Why a staging bucket?** When you deploy an agent, Vertex AI needs to upload your Python code somewhere before creating the managed endpoint. This GCS bucket is that temporary staging area. It must be in the same region as your Vertex AI deployment.

**Create the bucket first** (run once in terminal):
```bash
gsutil mb -l us-central1 gs://your-project-id-agent-staging
```

```python
vertexai.init(
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
    staging_bucket=STAGING_BUCKET,
)
```

**`vertexai.init()`** — Initializes the Vertex AI SDK. Must be called before any Vertex AI operations. Sets the default project, region, and staging bucket for all subsequent calls.

```python
adk_app = reasoning_engines.AdkApp(
    agent=host_agent,
    enable_tracing=False,
)
```

**`AdkApp`** — A wrapper that packages your ADK agent for deployment. Vertex AI doesn't understand raw ADK agents — it needs them wrapped in `AdkApp` first.

**`enable_tracing=False`** — Distributed tracing records every step of every agent call for debugging. Disabled here to keep costs down. Enable it in production if you need to debug issues.

```python
remote_agent = reasoning_engines.ReasoningEngine.create(
    adk_app,
    requirements=[
        "google-adk==1.9.0",
        "google-genai",
        "cloudpickle>=3.0.0",
    ],
    display_name="Competitive Intelligence Agent",
    description="...",
)
```

**`ReasoningEngine.create()`** — Deploys your agent to Vertex AI. This takes 3-5 minutes. Internally it:
1. Serializes your Python agent code
2. Uploads it to the staging bucket
3. Creates a Docker container with your code + requirements
4. Starts a managed endpoint

**`requirements=[...]`** — Python packages to install in the cloud environment. Must include everything your agent needs to run. `cloudpickle` is needed for serializing the agent code.

**`display_name`** — A human-readable label shown in the GCP Console.

```python
print(f"   Resource name : {remote_agent.resource_name}")
```

**`remote_agent.resource_name`** — The full path to your deployed agent in GCP:
```
projects/your-project/locations/us-central1/reasoningEngines/1234567890
```
**Save this string.** You need it to call the agent from the Flask app or future notebook sessions.

---

## Section 11 — Query the Deployed Agent

```python
RESOURCE_NAME = remote_agent.resource_name

client = vertexai.Client(
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
)

deployed_agent = client.agent_engines.get(name=RESOURCE_NAME)
```

**Why `vertexai.Client` and not `reasoning_engines.ReasoningEngine()`?**

`vertexai.Client` is the modern API. It correctly binds all streaming methods (`stream_query`, `create_session`, etc.) as Python callable methods on the returned object. The older `ReasoningEngine()` approach retrieves the resource metadata but fails to bind streaming methods properly, causing `AttributeError`.

**`client.agent_engines.get(name=RESOURCE_NAME)`** — Retrieves your deployed agent by its resource name and returns an `AgentEngine` object with all methods properly attached.

```python
session = deployed_agent.create_session(user_id="prod_user")
print(f"Session created: {session['id']}")
```

**`create_session()`** — Creates a new conversation session on Vertex AI's managed session storage. Unlike `InMemorySessionService`, this session persists in the cloud even after your notebook closes.

**`session['id']`** — The session dictionary contains an `id` key with the unique session identifier. You pass this to every query so the agent knows which conversation to continue.

```python
for event in deployed_agent.stream_query(
    user_id="prod_user",
    session_id=session["id"],
    message="Analyze competitor: OpenAI",
):
    content = event.get("content", {})
    if content:
        for part in content.get("parts", []):
            text = part.get("text", "")
            if text:
                final_report += text
                print(text, end="", flush=True)
```

**`stream_query()`** — Sends a message to the deployed agent and streams the response back as a series of event dictionaries (unlike the local runner which yielded event objects).

**`event.get("content", {})`** — Safely reads the `"content"` key from the event dictionary. If `"content"` doesn't exist, returns an empty dict `{}` instead of raising a `KeyError`. The `.get()` method is safer than `event["content"]`.

**`content.get("parts", [])`** — Gets the list of parts from the content. Returns empty list `[]` if not present — safe to loop over even if empty.

**`print(text, end="", flush=True)`** — Prints text as it streams in, without adding a newline after each chunk (`end=""`). `flush=True` forces immediate display instead of buffering — you see the report appear word by word in real time.

---

## End-to-End Flow Summary

```
Notebook run
    │
    ├── Section 1-3:  Install packages, configure GCP, import libraries
    │
    ├── Section 4:    Define 4 agents (market scanner, sentiment, pricing, report)
    │
    ├── Section 5:    Wire them into a SequentialAgent pipeline
    │
    ├── Section 6:    Create local Runner with InMemory services
    │
    ├── Section 7-9:  Test locally — single agent, then full pipeline
    │
    ├── Section 10:   Deploy full pipeline to Vertex AI Agent Engine
    │                 (3-5 min — creates a permanent cloud endpoint)
    │
    └── Section 11:   Query the deployed endpoint
                      → stream_query returns live results from the cloud
```

---

## Common Errors and What They Mean

| Error | Cause | Fix |
|---|---|---|
| `ValueError: Session not found` | ADK 1.9.0 needs explicit session creation | The `get_session → create_session` block in Section 7 handles this |
| `DefaultCredentialsError` | GCP auth not set up | Run `auth.authenticate_user()` in Section 2 (Colab) or `gcloud auth application-default login` (local/Codespaces) |
| `ValueError: Please provide a staging_bucket` | Missing GCS bucket in `vertexai.init()` | Set `STAGING_BUCKET` and pass it to `vertexai.init()` in Section 10 |
| `AttributeError: stream_query` | Using old `ReasoningEngine()` API | Use `vertexai.Client` + `agent_engines.get()` as in Section 11 |
| `gcloud: command not found` | gcloud SDK not installed | Run `curl https://sdk.cloud.google.com \| bash` then `exec -l $SHELL` |
