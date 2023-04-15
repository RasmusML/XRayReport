import logging
import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def prepare_for_evaluation(model, dataset, token2id, id2token, device=None, max_length=200, log_every=15, early_exit=-1):
    references = []
    candidates = []

    for i, (xray, token_ids, _) in enumerate(dataset):
        
        if i == early_exit:
            break

        if device:
            xray = xray.to(device)
        
        if log_every > 0 and i % log_every == 0:
            total_size = len(dataset) if early_exit == -1 else min(early_exit, len(dataset))
            logging.info(f"passing sample {i}/{total_size}")
        
        target = [id2token[token] for token in token_ids[1:-1]]
        references.append([target])

        #generated_ids = beam_search(model, token2id, beam_width=3, max_length=max_length)[0]
        generated_ids = greedy_search(model, xray, token2id, max_length=max_length)
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


def compute_bleu(model, dataset, token2id, id2token, device=None, max_length=200, early_exit=-1):
    references, candidates = prepare_for_evaluation(model, dataset, token2id, id2token, device=device, max_length=max_length, early_exit=early_exit)
    bleu = bleu_score(references, candidates)
    return bleu


def greedy_search(model, xray, token2id, max_length=200):
    start_token = token2id["<START>"]
    token_ids = [start_token]

    emit = model.cached_emitter(xray)

    for _ in range(1, max_length):
        t_token_ids = torch.tensor(token_ids).to(xray.device)
        scores = emit(t_token_ids)
        token_id = torch.argmax(scores).detach().cpu().item()
        token_ids.append(token_id)

        if token_id == token2id["<END>"]:
            break

    return token_ids


def prob_sample(model, xray, token2id, max_length=200):
    start_token = token2id["<START>"]
    token_ids = [start_token]

    emit = model.cached_emitter(xray)

    for _ in range(1, max_length):
        t_token_ids = torch.tensor(token_ids).to(xray.device)
        scores = emit(t_token_ids)
            
        p = F.softmax(scores, dim=-1).detach().cpu().numpy().astype(np.float64)
        p /= np.sum(p)

        token_id = torch.tensor(np.random.choice(len(p), p=p))
        token_ids.append(token_id.item())

        if token_id == token2id["<END>"]:
            break

    return token_ids


def beam_search(model, xray, token2id, beam_width=5, max_length=100):
    start_token = token2id["<START>"]

    beams = [(0, [start_token]) for _ in range(beam_width)] # (score, tokens)
    done = []

    emit = model.cached_emitter(xray)

    scores = emit(torch.tensor([start_token]))
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
            scores = emit(torch.tensor(token_ids))
            all_scores[idx] += scores

        _, top_idx = all_scores.flatten().topk(beam_width)

        beam_ids = top_idx // vocab_size
        token_ids = top_idx % vocab_size

        for idx in range(beam_width):
            beam_id, token_id = beam_ids[idx], token_ids[idx]
            new_score = all_scores[beam_id, token_id].item()
            new_token_ids = all_token_ids[beam_id] + [token_id.item()]
            beams[idx] = (new_score, new_token_ids)

        done += [(score, token_ids) for (score, token_ids) in beams if token_ids[-1] == token2id["<END>"]]
        beams = [(score, token_ids) for (score, token_ids) in beams if token_ids[-1] != token2id["<END>"]]

    done.extend(beams)

    return done


def compute_most_similiar(token_id, embeddings, n_most_similar=1, eps=1e-5):
    assert n_most_similar > 0

    cosine_simularities = torch.zeros(embeddings.shape[0])
    embed = embeddings[token_id]

    for i, other in enumerate(embeddings):
        embed_eps = embed.double() + eps
        other_eps = other.double() + eps
        cosine_simularities[i] = torch.dot(embed_eps, other_eps) / (torch.linalg.vector_norm(embed_eps) * torch.linalg.vector_norm(other_eps))

    values, indices = cosine_simularities.topk(n_most_similar + 1)

    return [(id.item(), sim.item()) for id, sim in zip(indices[1:], values[1:])] # skip itself


def similiar_convert(most_similiar, id2token):
    return [(id2token[id], sim) for id, sim in most_similiar]