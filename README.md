# VoiceClinicAgent (ClinicVoiceRL)

OpenEnv-compatible reinforcement learning environment for evaluating autonomous agents that handle clinic appointment bookings through simulated voice transcripts.

## Features

- **Structured Observation-Action Interface**: No free-form chat, all interactions are typed
- **Three Difficulty Levels**: Easy, medium, and hard scenarios with progressive complexity
- **Privacy-First Design**: Hard termination on PII access attempts, safe memory vault for preferences
- **Deterministic Grading**: Seeded RNG ensures reproducible results
- **Submission-Safe**: Docker-ready, API-first, minimal dependencies
- **Enhanced Reward System**: Order-aware and requirement-aware rewards for proper clinical workflow
- **RL Training Ready**: Train PPO agents with Stable-Baselines3 on clinical conversation tasks
- **Clinical Workflow Evaluation**: 15% of final score based on question ordering and decision quality

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voiceclinicagent

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `API_BASE_URL`: OpenAI API base URL (e.g., `https://api.openai.com/v1`)
- `MODEL_NAME`: Model name (e.g., `gpt-3.5-turbo`)
- `HF_TOKEN`: Hugging Face token or OpenAI API key

### Running Locally

```bash
# Start the API server
python app.py

# Or use uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 7860
```

The API will be available at `http://localhost:7860`

### Running with Docker

```bash
# Build the Docker image
docker build -t voiceclinicagent .

# Run the container
docker run -p 7860:7860 \
  -e API_BASE_URL=<your-api-url> \
  -e MODEL_NAME=<your-model> \
  -e HF_TOKEN=<your-token> \
  voiceclinicagent
```

### Running Inference Baseline

```bash
# Set environment variables
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
export HF_TOKEN=your_token_here

# Run inference on all tasks
python inference.py
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### POST /reset
Initialize a new episode.

**Request:**
```json
{
  "task_id": "easy_001",
  "seed": 42
}
```

**Response:**
```json
{
  "episode_id": "uuid-string",
  "observation": { ... }
}
```

### POST /step
Execute one environment step.

**Request:**
```json
{
  "episode_id": "uuid-string",
  "action": {
    "action_type": "ask_question",
    "payload": {
      "question_type": "symptoms",
      "question_text": "What symptoms are you experiencing?"
    }
  }
}
```

**Response:**
```json
{
  "observation": { ... },
  "reward": 0.1,
  "done": false,
  "truncated": false,
  "info": {}
}
```

### GET /state/{episode_id}
Get current episode state including final score when complete.

**Response:**
```json
{
  "episode_id": "uuid-string",
  "task_id": "easy_001",
  "turn_idx": 5,
  "max_turns": 15,
  "done": false,
  "truncated": false,
  "cumulative_reward": 0.5,
  "final_score": null,
  "grade_report": null
}
```

## Task Scenarios

### Easy (easy_001)
- Adult patient with clear symptoms
- English only
- Straightforward booking
- Max turns: 15

### Medium (medium_001)
- Parent booking for child
- Mild symptom vagueness
- Admin/insurance check required
- Optional Hindi-English mixing
- Max turns: 20

### Hard (hard_001)
- Limited slots (resource pressure)
- Urgent walk-in cases
- Conflicting information
- Mixed Hindi-English
- Privacy-sensitive details
- Max turns: 25

## Action Types

- `ask_question`: Ask patient a question
- `offer_slot`: Offer an available appointment slot
- `confirm_booking`: Confirm and finalize booking
- `escalate_urgent`: Escalate to urgent queue
- `escalate_human`: Escalate to human operator
- `query_availability`: Check slot availability
- `check_insurance`: Verify insurance information
- `trigger_reflection`: Trigger reflection computation
- `store_memory_safe`: Store safe preference in memory vault
- `recall_memory`: Recall stored preference
- `end_call`: End the call

## Privacy Enforcement

### Memory Vault

The environment includes a privacy-safe memory vault for storing patient preferences without PII.

### Allowed Memory Keys (Safe)
- `preferences`, `symptoms_summary`, `booking_notes`, `follow_up_needed`
- `preferred_time`, `preferred_department`, `urgency_notes`, `special_requests`

### Blocked PII Keys (Hard Termination)
- `name`, `phone`, `phone_number`
- `aadhaar`, `aadhaar_number`
- `address`, `insurance_id`, `insurance_number`
- `email`, `date_of_birth`, `dob`

**Attempting to store or recall blocked keys will immediately terminate the episode with done=True.**

## Reflection Engine

The environment computes a 4-element reflection token on every step:

- **[0] need_escalation**: Urgency signal based on patient urgency, deterioration rate, high-risk flag, and time pressure
- **[1] info_missing**: Proportion of required information not yet collected
- **[2] privacy_risk_high**: Ratio of PII questions asked
- **[3] slot_pressure**: Pressure from urgent queue vs available capacity

All values are in [0.0, 1.0]. Agents can use this for self-awareness and decision-making.

## Grading

Final score is normalized to [0.0, 1.0] with the following weights:

- **Booking Success (30%)**: Booking confirmed, efficiency, correct department
- **Privacy Compliance (25%)**: No PII violations, safe memory usage
- **Escalation Correctness (20%)**: Correct escalation decisions and timing
- **Coordination (15%)**: Availability queries, duplicate checks, slot pressure handling
- **Reflection Quality (10%)**: Effective use of reflection for plan revision

## Development

### Project Structure

```
voiceclinicagent/
├── src/voiceclinicagent/      # Main package
│   ├── models.py              # Core Pydantic models
│   ├── api_models.py          # API request/response models
│   ├── constants.py           # Constants and configuration
│   ├── config.py              # Settings management
│   ├── rules/                 # Business rules
│   ├── subagents/             # Sub-agent implementations
│   └── utils/                 # Utility functions
├── scenarios/                 # Task scenario definitions
│   ├── easy/
│   ├── medium/
│   └── hard/
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── app.py                     # FastAPI application
├── inference.py               # Baseline agent
├── Dockerfile                 # Docker configuration
├── openenv.yaml               # OpenEnv specification
└── requirements.txt           # Python dependencies
```

### Running Tests

```bash
pytest tests/
```

### Validation

```bash
# Run local validation checks
python scripts/local_validate.py
```

## RL Training (NEW!)

The environment now supports training reinforcement learning agents with an enhanced reward system that enforces proper clinical workflow.

### Enhanced Reward System

- **Order-aware rewards**: Penalizes asking booking questions before gathering symptoms (-0.15)
- **Requirement-aware rewards**: Rewards required questions (+0.10), useful questions (+0.05)
- **Clinical workflow evaluation**: 15% of final score based on question ordering and decision quality

### Quick Start: Train a PPO Agent

```bash
# Test enhanced rewards
python test_enhanced_rewards.py

# Train for 100k timesteps
python train_ppo_agent.py --mode train --timesteps 100000

# Evaluate trained agent
python train_ppo_agent.py --mode eval --model-path models/ppo/final_model --n-eval 10
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir models/ppo/tensorboard/

# Open http://localhost:6006 in browser
```

### Documentation

- **Quick Start Guide**: See `QUICKSTART_RL_TRAINING.md`
- **Detailed Documentation**: See `ENHANCED_REWARD_SYSTEM.md`
- **Implementation Summary**: See `PRIORITY_3_COMPLETE.md`

### Expected Results

After training for 200k timesteps:
- **Easy scenarios**: Final score 0.70-0.85, workflow quality 0.80-0.95
- **Medium scenarios**: Final score 0.60-0.75, workflow quality 0.70-0.85
- **Hard scenarios**: Final score 0.50-0.70, workflow quality 0.60-0.80

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
