import asyncio
import argparse
from typing import Optional

# --- Import process functions from each pipeline step ---
from pipeline_steps.A_index_generation.code import process as process_A
from pipeline_steps.B_index_pruning.code import process as process_B
from pipeline_steps.C_update_index_metadata.code import process as process_C
from pipeline_steps.D_generate_issues_and_resolutions.code import process as process_D
from pipeline_steps.E_cluster_intents.code import process as process_E
from pipeline_steps.F_cluster_issue_kbs.code import process as process_F
from pipeline_steps.G_generate_issue_kbs.code import process as process_G
from pipeline_steps.H_generate_info_kbs.code import process as process_H
from pipeline_steps.I_generate_tools.code import process as process_I
from pipeline_steps.J_generate_qm_parameters.code import process as process_J
from pipeline_steps.K_simulate_conversations.code import process as process_K
from pipeline_steps.M_rediscover_intents.code import process as process_M
from pipeline_steps.N_create_task_specific_datasets.code import process as process_N

def parse_args():
    parser = argparse.ArgumentParser(description='Run the CXM Benchmark Pipeline')
    
    # Required arguments
    parser.add_argument('--brand_name', type=str, required=True,
                      help='Name of the brand')
    parser.add_argument('--brand_overview', type=str, required=True,
                      help='Detailed description of the brand')
    
    # Optional arguments with defaults
    parser.add_argument('--kb_lang', type=str, default='en',
                      help='Language for knowledge base generation (default: en)')
    parser.add_argument('--conv_lang', type=str, default=None,
                      help='Language for conversation simulation (defaults to kb_lang if not specified)')
    
    # Pipeline parameters
    parser.add_argument('--cluster_issue_kbs_concurrency', type=int, default=5,
                      help='Concurrency for issue KB clustering (default: 5)')
    parser.add_argument('--cluster_intents_scarcity_threshold', type=int, default=3,
                      help='Scarcity threshold for intent clustering (default: 3)')
    parser.add_argument('--cluster_intents_compression_frequency', type=int, default=500,
                      help='Compression frequency for intent clustering (default: 500)')
    parser.add_argument('--qm_n_semaphore', type=int, default=10,
                      help='Number of semaphores for QM parameter generation (default: 10)')
    parser.add_argument('--sim_n_semaphore', type=int, default=10,
                      help='Number of semaphores for conversation simulation (default: 10)')
    parser.add_argument('--sim_n_convs', type=int, default=1000,
                      help='Number of conversations to simulate (default: 1000)')
    parser.add_argument('--sim_max_turns', type=int, default=80,
                      help='Maximum turns per conversation (default: 80)')
    parser.add_argument('--sim_get_new_personas', type=bool, default=True,
                      help='Whether to generate new personas for simulation (default: True)')
    
    return parser.parse_args()

async def run_pipeline(
    brand_name: str,
    brand_overview: str,
    kb_lang: str = 'en',
    conv_lang: Optional[str] = None,
    cluster_issue_kbs_concurrency: int = 5,
    cluster_intents_scarcity_threshold: int = 3,
    cluster_intents_compression_frequency: int = 500,
    qm_n_semaphore: int = 10,
    sim_n_semaphore: int = 10,
    sim_n_convs: int = 1000,
    sim_max_turns: int = 80,
    sim_get_new_personas: bool = True
):
    """
    Run the complete CXM Benchmark Pipeline with the given configuration.
    
    Args:
        brand_name: Name of the brand
        brand_overview: Detailed description of the brand
        kb_lang: Language for knowledge base generation
        conv_lang: Language for conversation simulation (defaults to kb_lang if not specified)
        cluster_issue_kbs_concurrency: Concurrency for issue KB clustering
        cluster_intents_scarcity_threshold: Scarcity threshold for intent clustering
        cluster_intents_compression_frequency: Compression frequency for intent clustering
        qm_n_semaphore: Number of semaphores for QM parameter generation
        sim_n_semaphore: Number of semaphores for conversation simulation
        sim_n_convs: Number of conversations to simulate
        sim_max_turns: Maximum turns per conversation
        sim_get_new_personas: Whether to generate new personas for simulation
    """
    # Set conversation language to KB language if not specified
    if conv_lang is None:
        conv_lang = kb_lang

    print(f"Starting pipeline for brand: {brand_name}")
    print(f"Knowledge base language: {kb_lang}")
    print(f"Conversation language: {conv_lang}")

    print("\nRunning A_index_generation...")
    await process_A(brand_name, brand_overview)

    print("\nRunning B_index_pruning...")
    await process_B(brand_name)

    print("\nRunning C_update_index_metadata...")
    await process_C(brand_name)

    print("\nRunning D_generate_issues_and_resolutions...")
    await process_D(brand_name)

    print("\nRunning E_cluster_intents...")
    await process_E(
        brand_name,
        scarcity_threshold=cluster_intents_scarcity_threshold,
        compression_frequency=cluster_intents_compression_frequency
    )

    print("\nRunning F_cluster_issue_kbs...")
    await process_F(brand_name, concurrency=cluster_issue_kbs_concurrency)

    print("\nRunning G_generate_issue_kbs...")
    await process_G(brand_name, brand_overview, kb_lang)

    print("\nRunning H_generate_info_kbs...")
    await process_H(brand_name, brand_overview, kb_lang)

    print("\nRunning I_generate_tools...")
    await process_I(brand_name)

    print("\nRunning J_generate_qm_parameters...")
    await process_J(brand_name, qm_n_semaphore)

    print("\nRunning K_simulate_conversations...")
    await process_K(
        brand_name=brand_name,
        brand_description=brand_overview,
        n_convs=sim_n_convs,
        n_semaphores=sim_n_semaphore,
        max_turns=sim_max_turns,
        get_new_personas=sim_get_new_personas,
        conv_lang=conv_lang
    )

    print("\nRunning M_rediscover_intents...")
    await process_M(brand_name)

    print("\nRunning N_create_task_specific_datasets...")
    process_N(brand_name)

    print("\nPipeline complete!")

async def main():
    args = parse_args()
    await run_pipeline(
        brand_name=args.brand_name,
        brand_overview=args.brand_overview,
        kb_lang=args.kb_lang,
        conv_lang=args.conv_lang,
        cluster_issue_kbs_concurrency=args.cluster_issue_kbs_concurrency,
        cluster_intents_scarcity_threshold=args.cluster_intents_scarcity_threshold,
        cluster_intents_compression_frequency=args.cluster_intents_compression_frequency,
        qm_n_semaphore=args.qm_n_semaphore,
        sim_n_semaphore=args.sim_n_semaphore,
        sim_n_convs=args.sim_n_convs,
        sim_max_turns=args.sim_max_turns,
        sim_get_new_personas=args.sim_get_new_personas
    )

if __name__ == "__main__":
    asyncio.run(main()) 