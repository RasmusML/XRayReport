import logging
import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def prepare_for_evaluation(emit_wrap, model, dataset, token2id, id2token, max_length=200, log_every=5):
    references = []
    candidates = []

    for i, (xray, token_ids, _) in enumerate(dataset):
        
        if log_every > 0 and i % log_every == 0:
            logging.info(f"sample {i}")
        
        #emit = playground_emit(model, x)
        #emit = vit_emit(model, x)
        emit = emit_wrap(model, xray)

        target = [id2token[token] for token in token_ids[1:-1]]
        references.append([target])

        #generated_ids = beam_search(emit, token2id, beam_width=1, max_length=max_length)[0]
        generated_ids = greedy_search(emit, token2id, max_length=max_length)
        generated = [id2token[token] for token in generated_ids[1:-1]]
        candidates.append(generated)

    return references, candidates


# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
def bleu_score(references, candidates):
    weights = [
        (1,0,0,0),
        (0.5,0.5,0,0),
        (0.33,0.33,0.33,0),
        (0.25,0.25,0.25,0.25)
    ]
    return corpus_bleu(references, candidates, weights=weights)


def greedy_search(emit_fn, token2id, max_length=100):
    start_token = token2id["[START]"]
    token_ids = [start_token]

    for _ in range(1, max_length):
        scores = emit_fn(torch.tensor(token_ids))
        token_id = torch.argmax(scores).item()
        token_ids.append(token_id)

        if token_id == token2id["[END]"]:
            break

    return token_ids


def prob_sample(emit_fn, token2id, max_length=100):
    start_token = token2id["[START]"]
    token_ids = [start_token]

    for _ in range(1, max_length):
        scores = emit_fn(torch.tensor(token_ids))
            
        p = F.softmax(scores, dim=-1).detach().numpy().astype(np.float64)
        p /= np.sum(p)

        token_id = torch.tensor(np.random.choice(len(p), p=p))
        token_ids.append(token_id.item())

        if token_id == token2id["[END]"]:
            break

    return token_ids


def beam_search(emit_fn, token2id, beam_width=5, max_length=100):
    start_token = token2id["[START]"]

    beams = [(0, [start_token]) for _ in range(beam_width)] # (score, tokens)
    done = []

    scores = emit_fn(torch.tensor([start_token]))
    _, top_idx = scores.topk(beam_width)
    for idx, (score, token_ids) in enumerate(beams):
        top_id = top_idx[idx]
        beams[idx] = (score + scores[top_id].item(), token_ids + [top_id.item()])

    vocab_size = scores.shape[-1]

    for _ in range(2, max_length):
        beam_width = len(beams)

        if beam_width == 0:
            break
        
        all_token_ids = [token_ids for _, token_ids in beams]
        all_scores = torch.tensor([score for score, _ in beams]).reshape((beam_width, 1))
        all_scores = all_scores.expand((beam_width, vocab_size)).clone()

        for idx, (score, token_ids) in enumerate(beams):
            scores = emit_fn(torch.tensor(token_ids))
            all_scores[idx] += scores

        _, top_idx = all_scores.flatten().topk(beam_width)

        beam_ids = top_idx // vocab_size
        token_ids = top_idx % vocab_size

        for idx in range(beam_width):
            beam_id, token_id = beam_ids[idx], token_ids[idx]
            new_score = all_scores[beam_id, token_id].item()
            new_token_ids = all_token_ids[beam_id] + [token_id.item()]
            beams[idx] = (new_score, new_token_ids)

        done += [(score, token_ids) for (score, token_ids) in beams if token_ids[-1] == token2id["[END]"]]
        beams = [(score, token_ids) for (score, token_ids) in beams if token_ids[-1] != token2id["[END]"]]

    done.extend(beams)

    return done