"""Triplet Loss with online mining, implemented manually in TensorFlow.

Implements the FaceNet triplet loss  L = max(||f(a)-f(p)||^2 - ||f(a)-f(n)||^2 + margin, 0)
with two online-mining strategies computed inside each mini-batch:

  * "semi-hard" — for every anchor-positive pair pick the semi-hard negative
    (further than the positive but within the margin), falling back to the
    hardest negative when no semi-hard one exists (Schroff et al., 2015).
  * "hard"      — batch-hard mining (hardest positive vs hardest negative
    per anchor, Hermans et al., 2017).

No external loss libraries are used.
"""

import tensorflow as tf

import config


def _pairwise_squared_distances(embeddings):
    """||e_i - e_j||^2 for all pairs. Shape: (B, B)."""
    dot = tf.matmul(embeddings, embeddings, transpose_b=True)
    sq_norms = tf.linalg.diag_part(dot)
    d2 = tf.expand_dims(sq_norms, 1) - 2.0 * dot + tf.expand_dims(sq_norms, 0)
    return tf.maximum(d2, 0.0)


def _masked_minimum(data, mask, dim=1):
    axis_max = tf.reduce_max(data, dim, keepdims=True)
    return tf.reduce_min((data - axis_max) * mask, dim, keepdims=True) + axis_max


def _masked_maximum(data, mask, dim=1):
    axis_min = tf.reduce_min(data, dim, keepdims=True)
    return tf.reduce_max((data - axis_min) * mask, dim, keepdims=True) + axis_min


def triplet_loss_semihard(labels, embeddings, margin=None):
    """Online semi-hard triplet mining over the batch."""
    margin = margin if margin is not None else config.TRIPLET_MARGIN
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1, 1])
    batch_size = tf.size(labels)

    pdist = _pairwise_squared_distances(embeddings)
    adjacency = tf.equal(labels, tf.transpose(labels))       # same identity
    adjacency_not = tf.logical_not(adjacency)

    # For every (anchor a, positive p) pair, look for negatives n with
    # d(a,n) > d(a,p): those are the semi-hard candidates.
    pdist_tile = tf.tile(pdist, [batch_size, 1])
    mask = tf.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.greater(pdist_tile, tf.reshape(tf.transpose(pdist), [-1, 1])))
    mask_final = tf.reshape(
        tf.greater(tf.reduce_sum(tf.cast(mask, tf.float32), 1, keepdims=True), 0.0),
        [batch_size, batch_size])
    mask_final = tf.transpose(mask_final)

    adjacency_not_f = tf.cast(adjacency_not, tf.float32)
    mask_f = tf.cast(mask, tf.float32)

    # closest negative that is still further than the positive (semi-hard)
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_tile, mask_f), [batch_size, batch_size])
    negatives_outside = tf.transpose(negatives_outside)
    # fallback: hardest negative overall
    negatives_inside = tf.tile(
        _masked_maximum(pdist, adjacency_not_f), [1, batch_size])

    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)
    loss_mat = margin + pdist - semi_hard_negatives

    mask_positives = (tf.cast(adjacency, tf.float32)
                      - tf.linalg.diag(tf.ones([batch_size])))
    num_positives = tf.reduce_sum(mask_positives)

    return tf.math.divide_no_nan(
        tf.reduce_sum(tf.maximum(loss_mat * mask_positives, 0.0)),
        num_positives)


def triplet_loss_hard(labels, embeddings, margin=None):
    """Batch-hard triplet mining over the batch."""
    margin = margin if margin is not None else config.TRIPLET_MARGIN
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1, 1])
    batch_size = tf.size(labels)

    pdist = _pairwise_squared_distances(embeddings)
    adjacency = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
    positives_mask = adjacency - tf.linalg.diag(tf.ones([batch_size]))
    negatives_mask = 1.0 - adjacency

    hardest_positive = tf.squeeze(_masked_maximum(pdist, positives_mask), 1)
    hardest_negative = tf.squeeze(_masked_minimum(pdist, negatives_mask), 1)

    loss = tf.maximum(hardest_positive - hardest_negative + margin, 0.0)
    return tf.reduce_mean(loss)


def triplet_loss(labels, embeddings):
    if config.TRIPLET_MINING == "hard":
        return triplet_loss_hard(labels, embeddings)
    return triplet_loss_semihard(labels, embeddings)


def triplet_accuracy(labels, embeddings):
    """Fraction of anchors whose hardest positive is closer than the
    nearest negative — a rank-1 verification accuracy inside the batch."""
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1, 1])
    batch_size = tf.size(labels)

    pdist = _pairwise_squared_distances(embeddings)
    adjacency = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
    positives_mask = adjacency - tf.linalg.diag(tf.ones([batch_size]))
    negatives_mask = 1.0 - adjacency

    hardest_positive = tf.squeeze(_masked_maximum(pdist, positives_mask), 1)
    hardest_negative = tf.squeeze(_masked_minimum(pdist, negatives_mask), 1)

    return tf.reduce_mean(
        tf.cast(hardest_positive < hardest_negative, tf.float32))
