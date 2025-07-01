from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bleurt import score

def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score between a reference and a hypothesis.

    Args:
        reference (str): The reference text.
        hypothesis (str): The hypothesis text.

    Returns:
        float: The BLEU score.
    """
    bleu = BLEU(max_ngram_order=1, effective_order=True)
    return bleu.sentence_score(hypothesis, [reference]).score

def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE score between a reference and a hypothesis.

    Args:
        reference (str): The reference text.
        hypothesis (str): The hypothesis text.

    Returns:
        dict: The ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)['rougeL'].fmeasure

def calculate_bleurt(reference, hypothesis):
    """
    Calculate BLEURT score between a reference and a hypothesis.

    Args:
        reference (str): The reference text.
        hypothesis (str): The hypothesis text.

    Returns:
        float: The BLEURT score.
    """
    checkpoint = "pretrains/BLEURT-20"
    bleurt_scorer = score.BleurtScorer(checkpoint)
    return bleurt_scorer.score(references=[reference], candidates=[hypothesis])[0]

if __name__ == "__main__":
    # Example usage
    ref = "១០ នាក់"
    hyp = "មនុស្ស ១០ នាក់"
    score_bleu = calculate_bleu(ref, hyp)
    score_rouge = calculate_rouge(ref, hyp)
    score_bleurt = calculate_bleurt(ref, hyp)
    
    print(f"BLEU score: {score_bleu:.2f}")
    print(f"ROUGE scores: {score_rouge:.2f}")
    print(f"BLEURT score: {score_bleurt:.2f}") # Better for Khmer language here