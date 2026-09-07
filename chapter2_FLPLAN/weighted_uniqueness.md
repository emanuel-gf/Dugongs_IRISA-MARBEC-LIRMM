# Weighted kNN Uniqueness Scoring

## What Is Uniqueness?

Uniqueness measures how **geometrically isolated** a sample is in the embedding
space.  Given a set of L2-normalised feature vectors, a sample scores high if
its nearest neighbours are far away — meaning it represents a visual pattern
that is rare or underrepresented in the dataset.

Formally, for sample $i$ with cosine distances
$d_{i,1} \leq d_{i,2} \leq \dots \leq d_{i,k}$ to its $k$ nearest neighbours:

$$
U_i = \sum_{j=1}^{k} w_j \cdot d_{i,j}
$$

where $w_j$ is the weight assigned to the $j$-th nearest neighbour.
Scores are normalised to $[0, 1]$ by dividing by the global maximum.

---

## Why Weighted Instead of Plain Mean?

The nearest neighbour carries the strongest signal about local isolation.
The 10th neighbour is much less informative — by rank 10, the distances are
already influenced by the global density of the space rather than the local
neighbourhood of the sample.

Applying **decaying weights** by rank ensures that the uniqueness score
reflects genuine local isolation rather than being diluted by distant,
less-informative neighbours.

---

## Decay Families

Three parametric decay families are supported.  All weights are normalised
to sum to 1 before computing the weighted mean.

### 1. Exponential Decay

$$w(i) = e^{-\lambda i}$$

The parameter $\lambda$ controls how fast the weights drop:

| $\lambda$ | Behaviour |
|---|---|
| 0.1 | Very slow decay — all neighbours weighted almost equally |
| 0.5 | Moderate decay — recommended default |
| 1.0 | Aggressive decay — only the nearest neighbour matters |

**Use when:** you want a smooth, tunable control over the influence of distant
neighbours.  The exponential family is the most common in machine learning
distance-based scoring.

### 2. Power Decay

$$w(i) = \frac{1}{i^p}$$

The parameter $p$ controls steepness:

| $p$ | Name | Behaviour |
|---|---|---|
| 0.5 | Square root | Very slow — distant neighbours still contribute |
| 1.0 | Harmonic | Moderate — $w = (1, 1/2, 1/3, \dots)$ |
| 2.0 | Squared | Fast — $w = (1, 1/4, 1/9, \dots)$ |

**Use when:** you want an interpretable, rank-based weighting with a natural
"harmonic series" feel.  The harmonic ($p=1$) is a principled default.

### 3. Linear Decay

$$w(i) = \frac{k + 1 - i}{k}$$

No free parameter.  The nearest neighbour always gets weight 1.0, the
furthest always gets weight $1/k$.

**Use when:** you want the simplest possible decay with no hyperparameter
to tune and a transparent, linear relationship between rank and weight.

---

![Decay families comparison](decay_families.png)

*Each panel shows normalised weights for ranks 1–12.  Left: exponential at
five values of λ.  Centre: power at five values of p.  Right: linear
overlaid with exponential (λ=0.3) and harmonic (p=1) for direct comparison.*

---

## Choosing k — The Number of Neighbours

### Why k Matters

$k$ defines the **radius of the local neighbourhood** used to assess
isolation.  It has two competing effects:

- **Too small** ($k=2$): uniqueness is noisy — a single nearby duplicate
  can mask a genuinely isolated sample.
- **Too large** ($k \approx N/2$): the neighbourhood spans half the dataset.
  Every sample's distances average out to roughly the same value, collapsing
  the score distribution and destroying discriminative power.

### The $\sqrt{N}$ Rule of Thumb

A well-known heuristic from non-parametric statistics is to keep:

$$k < \sqrt{N}$$

where $N$ is the number of samples.  The intuition is that $\sqrt{N}$
represents the scale at which local and global structure are balanced —
larger than that and you are measuring global density rather than local
isolation.

**Examples for your dataset:**

| Dataset size $N$ | $\sqrt{N}$ | Recommended $k$ |
|---|---|---|
| 200 images | 14 | 5–10 |
| 750 images | 27 | 10–15 |
| 2 755 images (dugong) | 52 | 10–20 |
| 10 000 images | 100 | 15–30 |

### Empirical Effect of k

The right panel below shows the **standard deviation of uniqueness scores**
as a function of $k$ for a synthetic dataset of $N=300$ samples.  Standard
deviation is a proxy for **discriminative power** — higher std means the
scores spread out more and better distinguish isolated from clustered samples.

The std peaks at small $k$ and then monotonically decreases.  Past
$k = \sqrt{N}$ (red dashed line), the curve enters the "over-smoothed
region" where increasing $k$ further compresses scores and degrades the
ability to rank samples by informativeness.

![Effect of k on score distribution](k_effect.png)

*Left: score histograms for different k values — note how larger k compresses
all scores toward the centre.  Right: score std vs k — discriminative power
drops sharply past $\sqrt{N}$.*

---

## Recommended Settings for ACLR on Aerial Dugong Imagery

For your dataset ($N \approx 2755$ source images):

```python
uniqueness_scores, ids = compute_uniqueness_field(
    dataset,
    embeddings_field = "full_embeddings",
    uniqueness_field = "uniqueness_score",
    k            = 15,           # well below sqrt(2755) ≈ 52
    decay        = "exponential",
    decay_param  = 0.5,          # moderate decay
    verbose      = True,
)
```

The goal is to select samples that are **scene-level diverse** — images
representing rare water conditions, lighting, dugong densities, or aerial
perspectives that are underrepresented in the current training pool.
Exponential decay with $\lambda=0.5$ and $k=15$ provides stable, locally
meaningful uniqueness scores without over-sensitivity to single outliers.

---

## API Reference

```python
compute_uniqueness_field(
    dataset,
    embeddings_field = "full_embeddings",  # DINOv3 embeddings field
    uniqueness_field = "uniqueness_score", # output field name
    k                = 10,                 # number of neighbours
    decay            = "exponential",      # "exponential" | "linear" | "power"
    decay_param      = 0.5,               # λ for exponential, p for power
    custom_weights   = None,              # override with a manual list
    save             = True,              # write scores to dataset
    verbose          = True,              # print progress
)
# returns: (uniqueness_scores: np.ndarray, sample_ids: list)
```