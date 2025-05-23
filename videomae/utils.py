
import numpy as np
from evaluate import load
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

def compute_captioning_metrics(predictions, references):
    

    assert len(predictions) == len(references), "Length mismatch between predictions and references."

    bleu = load("bleu")
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])

    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    meteor_scores = [single_meteor_score(ref.strip().split(), pred.strip().split()) for pred, ref in zip(predictions, references)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
    exact_match_acc = sum(exact_matches) / len(exact_matches)

    return {
        "BLEU": bleu_result['bleu'],
        "ROUGE-L": rouge_scores['rouge-l']['f'],
        "METEOR": avg_meteor,
        "Exact Match": exact_match_acc
    }





def compute_top_k(scores, labels, K=1):
    """Compute top-k accuracy for actions."""
    scores = np.array(scores)
    labels = np.array(labels)
    NUM_TEST_SEG = scores.shape[0]
    NUM_TEST_SEG = labels.shape[0]

    correct_count = 0
    for i in range(NUM_TEST_SEG):
        if int(labels[i]) in scores[i].argsort()[-K:]:
            correct_count += 1

    accuracy = 100.0 * float(correct_count) / NUM_TEST_SEG
    #logger.info('Top-%d: %.04f%%' % (K, accuracy))
    return accuracy