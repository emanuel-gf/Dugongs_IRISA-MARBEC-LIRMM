Optimal Transport 

The Earth Mover's Problem (intuition)
Imagine you have 716 piles of sand (NC images) and 2039 holes (WP images). Each pile and each hole has a fixed amount of sand (mass). The question OT asks is:

What is the cheapest way to move all the sand from the piles into the holes?

"Cheap" means minimizing the total mass × distance traveled. The coupling matrix is the solution: it tells you exactly how much sand flows from each pile to each hole.

ot.dist — the Cost Matrix
ot.dist(Xs, Xt) computes the pairwise ground cost between every source point and every target point. It is not a loss — it is an input to the optimization problem, encoding your prior belief about what makes two points similar or dissimilar.
Mathematically:

```math
M_{ij} = c(x_i^{NC}, x_j^{WP})
```

For metric='sqeuclidean':
```math
M_{ij} = \| \mathbf{x}_i^{NC} - \mathbf{x}_j^{WP} \|_2^2
```
In your case, $M_{ij}$

$M_{ij}$​ answers: *"how geometrically far apart are NC image ${ii}$
i and WP image jj
j in the 1024-dim DINOv2 embedding space?"* If two images are visually similar (both showing dugongs in shallow water), their embeddings will be close and MijM_{ij}
Mij​ will be small — meaning it is **cheap** to transport mass between them.

## optimization problem:

Given the cost matrix $M$, Optimal Transport (OT) solves:

$$
\min_{T \geq 0} \sum_{i,j} T_{ij} \cdot M_{ij}
$$

subject to the marginal constraints:

$$
\sum_j T_{ij} = a_i \quad \text{(all NC mass is shipped out)}
$$

$$
\sum_i T_{ij} = b_j \quad \text{(all WP mass is received)}
$$

where

$$
a_i = \frac{1}{716}, \qquad b_j = \frac{1}{2039}
$$

with uniform weights.

This is a **linear program** — the *Kantorovich formulation* of Optimal Transport.

The solution $T^*$ is the **optimal transport plan**, also called the **coupling matrix**.

## Coupling Matrix

The Coupling Matrix coupling_
``ot_transport.coupling_`` is $T^*$, the solution to the problem above.

T.shape = (716, 2039)

```txt
WP_0     [ WP_1      WP_2    ...              ]
NC_0     [ 0.00000   0.00000   0.00140   ...  ]
NC_1     [ 0.00000   0.00132   0.00000   ...  ]
NC_2     [ 0.00000   0.00000   0.00000   ...  ]
```

Each entry $T_{ij}$ is the amount of mass flowing from NC image $_i$ to WP image $_j$. Three key properties:

1. Row sums = source weights:
$
\sum_j T_{ij} = \frac{1}{716} \approx 0.00140
$

Every NC image ships all its mass out.

2. Column sums = target weights:
$
\sum_i T_{ij} = \frac{1}{2039} \approx 0.00049
$

Every WP image receives exactly its share.

3. Sparsity: In the exact EMD solution, at most $716+2039 -1$  entries are non-zero (a property of linear programs). Each NC image typically gets matched to only a handful of WP images — the ones most similar in embedding space.

## `transform()` to the target space

How transform() uses the coupling
Once you have $T^*$, transforming NC embeddings into WP space is a
barycentric projection:
```math
\tilde{x}_i^{NC} = \frac{\sum_j T_{ij} \cdot x_j^{WP}}{\sum_j T_{ij}}
```
In words: the adapted embedding for NC image ii
i is a
weighted average of WP embeddings, where the weights come from row ii
i of the coupling matrix. Images that received high transport mass from NC image ii
i contribute more to its new position.

```markdown
# This is essentially what transform() does internally:
nc_adapted = (coupling / coupling.sum(axis=1, keepdims=True)) @ wp_emb
# shape: (716, 1024)
```

So if NC image 5 (a dugong in clear water) gets matched heavily to WP images 12, 47, and 203 (also dugongs in clear water), its adapted embedding is pulled toward the centroid of those three WP embeddings.

---

## The Sinkhorn variant

Exact EMD is expensive ($O(n^3)$). Sinkhorn adds an **entropic regularization** term:

$$\min_{T \geq 0} \sum_{ij} T_{ij} M_{ij} - \varepsilon \sum_{ij} T_{ij} \log T_{ij}$$

The $-\varepsilon H(T)$ entropy term encourages $T$ to be **smoother and denser** (less sparse). The parameter `reg_e` controls this tradeoff:

- **Large `reg_e`** → very smooth/uniform coupling, fast convergence, but less faithful matching
- **Small `reg_e`** → closer to exact OT, sparser coupling, slower convergence

The Sinkhorn algorithm solves this via iterative row/column normalization, which is parallelizable and GPU-friendly — hence its popularity for large-scale problems like yours (716 × 2039).



```markdown
         ot.dist()              OT solver              transform()
                                                        
nc_emb ──────────┐         ┌──────────────┐        
                 │  cost   │              │coupling │
                 ├──── M ──►  min Σ T·M  ├──── T ──► T @ wp_emb/T.sum
                 │ (716,   │  s.t. margins│ (716,   │  = nc_adapted
wp_emb ──────────┘  2039)  │              │  2039)  ────────
                           └──────────────┘
         "how far apart      "what is the       "move NC embeddings
          are all pairs?"     cheapest flow?"    to WP positions"
```


## Optimal Transport Plan T*

It represents the same for either `ot.emd(a,b,M)` or `ot.da.SinkhornTransport().fit().coupling_`.

However, `ot.emd` is a linear solver with $O(n^3)$ for small dasets <5k. Sinkhorn produces an entropic regularization with an approximation of $T^*$ ,scaling well to large datasets.

## Cosine SImilarity 

$$ \text{similarity}(x, y) = \frac{x \cdot y}{|x| |y|} = \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2} \cdot \sqrt{\sum_{i=1}^n y_i^2}} $$

## Euclidean Distance:

Euclidean distance measures the straight-line distance in the vector space.
$$ d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2} $$

##  Mixeed approach

Combine SSIM and embedding metrics (e.g., Cosine Similarity):
$$ \text{HybridScore}(x, y) = \alpha \cdot \text{SSIM-Score}(x, y) + \beta \cdot \text{Embedding-Metric}(x, y) $$

• Weights (α, β): Control the contribution of visual metrics vs. embedding alignment.  • A higher α prioritizes raw visual features (useful early in modeling).
  • A higher β emphasizes semantic embedding-level relationships (useful post-tuning).
