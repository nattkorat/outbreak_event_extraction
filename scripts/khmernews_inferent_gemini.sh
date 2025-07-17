# This script runs inference on the Khmer news dataset using the Gemini model.
python -m inferent.run \
    --test_data "data/20250716_khmer_news_scraped.json" \
    --few_shot_samples "speedpp_templates.json" \
    --output_file "khmernews_inferent_gemini.jsonl" \
    --model "gemini"