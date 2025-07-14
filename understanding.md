# Gemini's Project Understanding

This document serves as my internal state tracker for the GeethAI project. It ensures I remain aligned with the project goals and our collaborative decisions.

## Current Project State (As of 2025-07-14)

**Overall Progress:** Approximately 10-15% complete.

**Current Phase:** We are actively in **Phase 1: Lyrics Model ("Kavi")**.

**Completed Work:**
- **Project Foundation:** A comprehensive `GEMINI.md` constitution has been established.
- **Version Control:** The project is initialized with Git, has a proper `.gitignore`, and is synced with the `main` branch on GitHub.
- **Data Pipeline (Kavi):** A full suite of initial scripts for the lyrics model's data preparation has been created. This includes scripts for downloading, transcribing, preprocessing, and assembling the dataset.
- **Initial Training Script (Kavi):** The `train_kavi.py` script exists, marking the first step into the model training part of the phase.

**Next Immediate Goals:**
1.  Flesh out and test the data pipeline scripts to ensure they produce a high-quality dataset.
2.  Implement and run the `train_kavi.py` script to get a baseline fine-tuned model.
3.  Begin scaffolding the FastAPI service to serve the "Kavi" model once it's trained.

**Future Phases (Not Started):**
- Phase 2: Audio Model ("Sangeethakartha")
- Phase 3: Singing Voice Synthesis (SVS)
- Phase 4: Integration

---
## Action Plan: `train_kavi.py` Development

This plan outlines the steps to create a robust and efficient training script for the "Kavi" lyrics model while the full dataset is being generated.

**Guiding Principles:**
- **Efficiency:** Use modern techniques to train faster and with less memory.
- **Testability:** Ensure the entire pipeline works with a small "dummy" dataset before committing to a full run.

**Technical Implementation Plan:**
1.  **Data Loading:**
    - The script will be modified to read the `lyrics_dataset.txt` file.
    - It will parse the file, splitting the text into individual song lyrics using the `\n\n---\n\n` separator.

2.  **Dummy Run Functionality:**
    - A command-line argument (e.g., `--dummy_run`) will be added.
    - If enabled, the script will only use the first 20-30 lyrics from the dataset. This allows for rapid testing of the entire training and evaluation process.

3.  **Efficient Fine-Tuning (LoRA):**
    - We will implement **Low-Rank Adaptation (LoRA)** using the Hugging Face `peft` (Parameter-Efficient Fine-Tuning) library.
    - This freezes the large pre-trained model and only trains small, efficient "adapter" layers, dramatically reducing training time and resource usage.

4.  **Hugging Face `Trainer`:**
    - The script will use the high-level `Trainer` API from the `transformers` library.
    - This will manage the training loop, batching, evaluation, and checkpointing automatically.
    - We will configure `TrainingArguments` to set hyperparameters like learning rate, batch size, and save strategies.

5.  **Model & Tokenizer:**
    - We will use a powerful base model like `gemma-2-9b-it` or a similar high-performing model suitable for fine-tuning.
    - The corresponding tokenizer will be used to prepare the data for the model.

This plan will result in a production-ready training script that we can confidently run on the full dataset once it's available.
