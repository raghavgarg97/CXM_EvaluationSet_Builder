# CXMArena Synthetic CXM Benchmark Pipeline

A modular, extensible pipeline for generating, clustering, and simulating customer support knowledge and conversations for brands, designed to create large-scale, realistic synthetic datasets for Customer Experience Management (CXM) research and benchmarking.

## Overview

This repository implements a comprehensive data generation pipeline that enables the creation of synthetic, brand-specific datasets that simulate real-world CXM scenarios, including knowledge base (KB) construction, issue and intent clustering, conversation simulation, and the derivation of task-specific benchmark datasets. The generated data supports rigorous evaluation of AI models on operational CXM tasks such as knowledge base refinement, intent prediction, agent quality adherence, article search, and multi-turn retrieval-augmented generation (RAG).

**Key Features:**
- **Brand Customization:** Easily instantiate the pipeline for any fictional or real brand by providing a brand name and description.
- **LLM-Driven Generation:** Leverages state-of-the-art LLMs (OpenAI, Google Vertex, etc.) for all generative and clustering steps.
- **Multi-Step Modular Pipeline:** Each step is a standalone Python module, enabling flexible experimentation and extension.
- **Task-Specific Dataset Creation:** Outputs labeled datasets for five core CXM tasks.
- **Concurrent and Scalable:** Supports async and batched processing for efficient large-scale data generation.

## Pipeline Structure

The pipeline is organized as a sequence of modular steps, each responsible for a specific aspect of the synthetic data generation process. The main steps are:

1. **A_index_generation:** Generate a hierarchical brand knowledge index (taxonomy) from a brand description.
2. **B_index_pruning:** Prune redundant or overlapping nodes from the index using embedding-based similarity and LLM verification.
3. **C_update_index_metadata:** Update some left out index nodes detailed metadata and properties.
4. **D_generate_issues_and_resolutions:** Generate customer issues and their resolutions grounded in the brand's KB.
5. **E_cluster_intents:** Cluster issues into higher-level intents using LLM-based clustering and verification.
6. **F_cluster_issue_kbs:** Further cluster and refine issue KBs for operational realism.
7. **G_generate_issue_kbs:** Generate detailed issue KB articles and customer personas for each cluster.
8. **H_generate_info_kbs:** Generate information KB articles for the brand, linking them to issues and resolutions.
9. **I_generate_tools:** Identify and define function-calling tools (APIs) required for issue resolution.
10. **J_generate_qm_parameters:** Generate and cluster agent quality management (QM) evaluation parameters.
11. **K_simulate_conversations:** Simulate multi-turn, persona-driven customer-agent conversations grounded in the KBs and tools.
12. **L_verify_qm_parameters:** (Optional) Validate and refine QM parameters using LLMs.
13. **M_rediscover_intents:** Rediscover and refine intent taxonomies from simulated conversations.
14. **N_create_task_specific_datasets:** Derive labeled datasets for each benchmark task (KB refinement, intent prediction, agent QM, article search, multi-turn RAG).

Each step is implemented as an async Python module with a `process` function, and intermediate outputs are checkpointed for inspection and reuse.

## Usage

### Prerequisites

- Python 3.8+
- Access to required LLM endpoints (see `configs/llm_config.json`)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Pipeline

To run the pipeline, use the main entrypoint script `run_pipeline.py`. This script takes configuration parameters for the brand name, description, and step-specific settings.

**Example:**
```bash
python run_pipeline.py --brand_name "TerraBloom" --brand_overview "A sustainable gardening brand..." --kb_lang "en"
```

This will:
- Generate and cluster KBs and issues for the specified brand
- Simulate conversations in the specified language
- Output labeled datasets for all benchmark tasks

### Configuration

The pipeline can be configured through command-line arguments or by modifying the default settings in `run_pipeline.py`:

- **Brand Settings:**
  - `brand_name`: Name of the brand
  - `brand_overview`: Detailed description of the brand
  - `kb_lang`: Language for knowledge base generation (default: "en")

- **Pipeline Parameters:**
  - `num_conversations`: Number of conversations to simulate
  - `max_turns`: Maximum turns per conversation
  - `concurrency`: Number of concurrent operations
  - `checkpoint_dir`: Directory for saving intermediate outputs

### Customization

- **Step Parameters:** Adjust concurrency, number of conversations, max turns, etc., as needed.
- **Partial Pipeline:** You can import and call individual step modules directly for experimentation.

### Outputs

- Intermediate and final outputs are saved under `./checkpoints/<STEP>/<brand_name>/`.
- Final labeled datasets for each benchmark task are produced in the last step (`N_create_task_specific_datasets`).

## Directory Structure

```
ccai_evaluation_sets/
├── pipeline_steps/           # Modular pipeline steps (A-N)
│   └── <step_name>/code.py   # Each step's main logic
├── utils/                    # LLM helpers, constants, misc utilities
├── configs/                  # LLM, brand, and persona configs
├── run_pipeline.py          # Main pipeline entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## License

Specify your license here (e.g., MIT, Apache 2.0, CC-BY-NC 4.0, etc.).

## Contact & Support

For questions, support, or collaboration, please contact the maintainers via your internal channel or raise an issue in this repository.

---

**Note:** This pipeline is intended for research and benchmarking purposes. The generated data is synthetic and does not contain real customer information.
