python -m inferent.run \
    --test_data "data/KH_100_samples.json" \
    --few_shot_samples "data/templates/speedpp_templates_with_start_index.json" \
    --output_file "gemini_kh_100_samples.jsonl" \
    --model "gemini"