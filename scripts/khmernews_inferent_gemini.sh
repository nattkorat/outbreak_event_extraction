# This script runs inference on the Khmer news dataset using the Gemini model.
# python -m inferent.run \
#     --test_data "data/20250716_khmer_news_scraped_v1.json" \
#     --few_shot_samples "speedpp_templates.json" \
#     --output_file "khmernews_inferent_gemini_v1.jsonl" \
#     --model "gemini"

# change template style
python -m inferent.run \
    --test_data "data/20250716_khmer_news_scraped_v2.json" \
    --few_shot_samples "speedpp_templates.json" \
    --output_file "khmernews_inferent_gemini_v2.jsonl" \
    --model "gemini"