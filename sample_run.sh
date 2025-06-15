#!/bin/bash

# Run the CXM Benchmark Pipeline for TerraBloom
python run_pipeline.py \
  --brand_name "TerraBloom" \
  --brand_overview "TerraBloom Botanicals is a luxury, artisanal skincare and wellness brand dedicated to harnessing the potent power of nature's finest ingredients. We meticulously source rare and organic botanicals from sustainable farms across the globe, crafting small-batch elixirs designed to rejuvenate the skin and soothe the soul. TerraBloom champions a holistic approach to beauty, believing that true radiance comes from a harmonious balance of mind, body, and spirit. Our eco-conscious packaging and ethical production practices reflect our deep commitment to preserving the planet. Indulge in the TerraBloom Botanicals experience, where ancient herbal wisdom meets modern green science to unveil your most luminous self." \
  --kb_lang "French" \
  --conv_lang "French" \
  --cluster_issue_kbs_concurrency 5 \
  --cluster_intents_scarcity_threshold 3 \
  --cluster_intents_compression_frequency 500 \
  --qm_n_semaphore 10 \
  --sim_n_semaphore 10 \
  --sim_n_convs 1000 \
  --sim_max_turns 80 \
  --sim_get_new_personas true 