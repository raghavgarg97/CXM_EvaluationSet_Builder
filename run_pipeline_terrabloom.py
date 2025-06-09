import asyncio

# --- Configurable global variables ---
brand_name = 'TerraBloom'
brand_overview = '''TerraBloom Botanicals is a luxury, artisanal skincare and wellness brand dedicated to harnessing the potent power of nature's finest ingredients. We meticulously source rare and organic botanicals from sustainable farms across the globe, crafting small-batch elixirs designed to rejuvenate the skin and soothe the soul. TerraBloom champions a holistic approach to beauty, believing that true radiance comes from a harmonious balance of mind, body, and spirit. Our eco-conscious packaging and ethical production practices reflect our deep commitment to preserving the planet. Indulge in the TerraBloom Botanicals experience, where ancient herbal wisdom meets modern green science to unveil your most luminous self.'''

# Optional/advanced config (step-specific)
kb_lang = 'French'
cluster_issue_kbs_concurrency = 5  # For F_cluster_issue_kbs
cluster_intents_scarcity_threshold = 3  # For E_cluster_intents
cluster_intents_compression_frequency = 500  # For E_cluster_intents
qm_n_semaphore = 10  # For J_generate_qm_parameters

sim_n_semaphore = 10 # For K_simulate_conversations
sim_n_convs = 1000  # For K_simulate_conversations
sim_max_turns = 80  # For K_simulate_conversations
sim_get_new_personas = True  # For K_simulate_conversations
conv_lang = 'French' # For K_simulate_conversations

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
from pipeline_steps.K_simulate_conversations.code import process as process_K  # Uncomment if needed
from pipeline_steps.M_rediscover_intents.code import process as process_M
from pipeline_steps.N_create_task_specific_datasets.code import process as process_N

async def main():
    print("Running A_index_generation...")
    await process_A(brand_name, brand_overview)

    print("Running B_index_pruning...")
    await process_B(brand_name)

    print("Running C_update_index_metadata...")
    await process_C(brand_name)

    print("Running D_generate_issues_and_resolutions...")
    await process_D(brand_name)

    print("Running E_cluster_intents...")
    await process_E(brand_name, scarcity_threshold=cluster_intents_scarcity_threshold, compression_frequency=cluster_intents_compression_frequency)

    print("Running F_cluster_issue_kbs...")
    await process_F(brand_name, concurrency=cluster_issue_kbs_concurrency)

    print("Running G_generate_issue_kbs...")
    await process_G(brand_name, brand_overview, kb_lang)

    print("Running H_generate_info_kbs...")
    await process_H(brand_name, brand_overview, kb_lang)

    print("Running I_generate_tools...")
    await process_I(brand_name)

    print("Running J_generate_qm_parameters...")
    await process_J(brand_name, qm_n_semaphore)

    print("Running K_simulate_conversations...")
    await process_K(
        brand_name=brand_name,
        brand_description=brand_overview,
        n_convs=sim_n_convs,
        n_semaphores=sim_n_semaphore,
        max_turns=sim_max_turns,
        get_new_personas=sim_get_new_personas,
        conv_lang=conv_lang
    )

    print("Running M_rediscover_intents...")
    await process_M(brand_name)

    print("Running N_create_task_specific_datasets...")
    process_N(brand_name)

    print("Pipeline complete!")

if __name__ == "__main__":
    asyncio.run(main()) 