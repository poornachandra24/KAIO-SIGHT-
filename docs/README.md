# Project Omni-Nav: Architectural Dossier

This directory contains the complete system design, architectural decisions, and performance logs for **Project Omni-Nav**, a high-throughput Physical AI agent trained on the AMD Instinct MI300X.

This is not just a model; it is a production-grade pipeline designed for 2026-tier AI development, demonstrating mastery over hardware-software co-design, data engineering at scale, and advanced VLM fine-tuning.

## Table of Contents

### 1. Architecture & Design
*   [**System Architecture**](./architecture/system_design.md)
    *   *High-level diagrams of the orchestration, data flow, and model architecture.*
*   [**Data Lifecycle**](./architecture/data_lifecycle.md)
    *   *A deep dive into the ingestion, ETL, training load, and inference stages.*
*   [**Hardware Optimization (MI300X)**](./architecture/hardware_optimization.md)
    *   *Specific strategies used to saturate the 192GB HBM3 memory and optimize for ROCm.*

### 2. Reports & Logs
*   [**Architectural Decision Log (ADR)**](./reports/decision_log.md)
    *   *A log of critical architectural decisions (e.g., Bfloat16, Map-Style datasets) and their outcomes.*
*   [**Challenges & Resolutions**](./reports/challenges_fixes.md)
    *   *A record of major technical hurdles (Broadcasting crashes, Empty Tensors) and how they were fixed.*
*   [**Data Audit**](./reports/data_audit.md)
    *   *Automated data integrity reports.*

### 3. Operational Guides
*   [**Data Setup**](./data_setup/data_sourcing.md)
    *   *Guide to data sourcing and downloading.*
*   [**Preprocessing**](./data_setup/preprocessing.md)
    *   *Details on the Turbo ETL pipeline.*
*   [**Training Pipeline**](./finetuning/training_pipeline.md)
    *   *How to run the training orchestration.*
*   [**Inference Flow**](./inference/inference_flow.md)
    *   *Guide to running inference.*
