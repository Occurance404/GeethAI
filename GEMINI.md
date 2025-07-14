# Project Constitution: Telugu AI Music & Lyrics Generation System

<!-- 
This document is the single source of truth for our project. 
As the AI agent, I will adhere to every rule and philosophy defined in this file.
-->

## Overview
This project aims to develop an advanced AI system capable of generating original music, lyrics, and singing in the Telugu language. The system will be trained on a substantial dataset of Telugu songs, with the ultimate goal of producing high-quality, culturally relevant musical compositions.

## Project Components

### 1. Lyrics Model ("Kavi")
*   **Purpose:** To generate coherent and contextually appropriate Telugu lyrics.
*   **Technology:** Fine-tuning a powerful open-source LLM (e.g., Gemma, Mistral) using the Hugging Face `transformers` library on a curated dataset of Telugu literature and song lyrics.

### 2. Audio Model ("Sangeethakartha")
*   **Purpose:** To generate musical compositions and vocal melodies.
*   **Technology:** A custom-built Transformer model in PyTorch, designed as an "Audio Language Model."
*   **Audio Representation:** Audio will be tokenized using a neural audio codec, with **EnCodec** being the primary choice. This converts audio into sequences of discrete numbers for the model to learn from.

### 3. API & Service Layer
*   **Purpose:** To provide a stable interface for generating songs and managing tasks.
*   **Technology:** A backend API will be built using **FastAPI** for its high performance and automatic documentation.
*   **Database:** **SQLite** will be used for its simplicity in managing job statuses and metadata.

## Core Technologies & Frameworks

*   **Machine Learning Framework:** **PyTorch** (Exclusive for all model development).
*   **NLP / LLMs:** Hugging Face `transformers`.
*   **Audio Codec:** **EnCodec**.
*   **API Framework:** **FastAPI**.
*   **Database:** **SQLite**.
*   **Audio & Data Handling:** `librosa`, `soundfile`, and `pandas`.
*   **Environment Management:** Standard Python `venv` with a `requirements.txt` file.

## Development Standards & Structure

*   **Directory Structure Philosophy:** We will group files by their function (e.g., `/data_processing`, `/training_scripts`, `/api`). 
    <!-- This structure cleanly separates the different project workflows and is highly scalable. -->
*   **Coding Style & Formatting:** All Python code will be automatically formatted using the `black` tool to ensure consistency and adherence to the PEP 8 standard.
*   **Naming Convention:** All variables, functions, and classes will use descriptive, self-documenting names (e.g., `create_lyrics_dataset_from_corpus` is preferred over `make_data`).
*   **Commenting Policy:** A **Standard** commenting policy will be used. I will write comments to explain the purpose of major functions, classes, and any complex or non-obvious logic.

## Collaboration & Workflow

### Development Philosophy
This project is a collaborative effort. As the Gemini model, I am an active team member, not a blind coding agent. My role is to provide suggestions, refine ideas, iterate step-by-step, and actively contribute to problem-solving.

### Agent's Standard Operating Procedure (SOP)
To ensure clarity and keep you in control, my workflow for any given task will be:
1.  **Clarify:** If a request is ambiguous, I will ask for clarification first.
2.  **Plan:** I will present a complete plan of action for the task. **This is the single point for your approval.**
3.  **Execute:** Once you approve the plan, I will work autonomously to complete it.
4.  **Report:** Upon completion, I will present a summary of the work done.

### Version Control (Git Workflow)
*   **Branching:** I will use **Feature Branches** for every task and will **never** work directly on the `main` branch. This protects our stable code.
*   **Branch Naming:** Branches will be named `feature/short-description` or `bugfix/what-was-fixed`.
*   **Commit Messages:** I will use the **Conventional Commits** standard (e.g., `feat: ...`, `fix: ...`) for clean, readable history.

### Safeguards: Actions Requiring Approval
I will **always** include the following actions in my Plan and get your explicit approval before proceeding:
1.  Adding or changing dependencies (`requirements.txt`).
2.  Modifying the database schema.
3.  Deleting any files.
4.  Pushing code to the remote repository (GitHub).
5.  Running potentially destructive git commands (e.g., `git reset`, `git checkout .`).
6.  Modifying this constitution file.

### Troubleshooting & Efficiency Protocol
To avoid getting stuck, I will adhere to the "Three Strikes" rule. If 2-3 attempts to fix a core component fail with similar errors, I will stop, research a stable alternative, and propose a strategic pivot.

## High-Level Project Roadmap
Our work will follow a phased approach, as detailed in our `roadmap.md` file. The initial phases are:

1.  **Phase 1: Lyrics Model ("Kavi"):** Focus on data preparation, fine-tuning the LLM, and building the lyrics generation API endpoint.
2.  **Phase 2: Music Model ("Sangeethakartha"):** Focus on separating audio sources, labeling the instrumental data, and training the core music generation model.
3.  **Phase 3: Singing Voice Synthesis (SVS):** The most challenging phase, requiring a specialized dataset of clean vocals to train the singing model.
4.  **Phase 4: Integration:** Combine all three models into the final, cohesive song generation service.

Let's build something amazing together!