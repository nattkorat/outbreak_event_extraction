python -m inferent.run \
    --test_data "data/KH_sample_10.json" \
    --few_shot_samples "data/templates/speedpp_templates_with_start_index.json" \
    --output_file "test_output.jsonl" \
    --model "gemini"
    