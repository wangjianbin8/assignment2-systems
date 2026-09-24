# CS336 Systems — Problem 3.2.1 Gradient Checkpointing

## Background: Activation Checkpointing and Recomputation

Without checkpointing, a Transformer with \(N\) sequential blocks keeps the residuals needed by backward for all \(N\) blocks alive at the same time, so peak activation memory grows as \(O(N)\).

`torch.utils.checkpoint.checkpoint` trades compute for memory. During the first forward pass, a checkpointed region keeps the region input (the checkpoint boundary activation) while suppressing the ordinary saved tensors inside that region. During backward, that saved input is used to recompute the region's forward pass, recreate the residuals needed for backward, run backward through the region, and then release those temporary residuals.

```text
normal training:
forward saves internal residuals
        ↓
backward directly consumes them

checkpointing:
forward saves region input
        ↓
internal residuals are discarded
        ↓
backward reaches region
        ↓
re-run forward from saved input
        ↓
recreate needed residuals
        ↓
run backward
```

The key distinction is:

- **checkpoint activation**: the input/boundary activation that lets us re-enter a region later;
- **residual / saved tensor**: an internal tensor needed to execute backward through operations inside the region.

## (a) Memory-Optimal Recursive Checkpointing

### Strategy

For the recursive checkpointing strategy used here, split the sequence of \(N\) Transformer blocks into approximately equal subregions and recursively checkpoint those regions until the leaves are very small.

For \(N=8\):

```text
                    B1...B8
                   /      \
               B1...B4   B5...B8
               /   \       /   \
            B1-B2 B3-B4 B5-B6 B7-B8
```

At any one stage of backward, we only need checkpoint boundary activations along the currently active recursive path, plus the residuals of the small region currently being recomputed. We do **not** keep the residuals for all \(N\) blocks simultaneously.

Conceptually, while descending toward the last block, the live checkpoint boundaries can look like:

```text
x0  -> input to a large region
x4  -> input to a smaller subregion
x6  -> input to a still smaller subregion
x7  -> input to the current block
```

The exact checkpoint boundaries can be created and released during recomputation; they do not all have to be permanently fixed by the initial forward pass.

### Peak activation memory

With balanced recursive splitting,

\[
N \rightarrow N/2 \rightarrow N/4 \rightarrow \cdots \rightarrow 1
\]

so the recursion depth is \(O(\log N)\). Only \(O(\log N)\) checkpoint levels need to be live at once, rather than \(O(N)\) blocks' residuals.

Therefore:

\[
\boxed{\text{Peak activation memory} = O(\log N)}
\]

instead of \(O(N)\).

### Compute complexity

At each recursion level, the total amount of recomputed block work across all subregions is still \(O(N)\):

```text
level 1:                 N
level 2:           N/2 + N/2 = N
level 3:   N/4 + N/4 + N/4 + N/4 = N
...
```

A balanced recursion has \(O(\log N)\) levels, so the total recomputation work is:

\[
N \times \log N.
\]

Thus:

\[
\boxed{\text{Compute} = O(N\log N)}
\]

(up to the ordinary \(O(N)\) forward/backward work, which does not change the highest-order term).

### Short code sketch

```python
from torch.utils.checkpoint import checkpoint


def recursive_checkpoint(blocks, x):
    # Base case: one block remains.
    if len(blocks) == 1:
        return blocks[0](x)

    # Split into two approximately equal regions.
    mid = len(blocks) // 2

    def run_region(inp):
        out = recursive_checkpoint(blocks[:mid], inp)
        out = recursive_checkpoint(blocks[mid:], out)
        return out

    return checkpoint(run_region, x, use_reentrant=False)
```

### Deliverable answer

A balanced recursive checkpointing strategy can reduce peak activation memory by recursively splitting the \(N\) Transformer blocks into approximately equal regions and nesting checkpoint calls until the leaves are small. During backward, only checkpoint boundary activations along the active recursion path and the currently recomputed region's residuals need to be live, giving \(O(\log N)\) peak activation memory instead of \(O(N)\). Each recursion level recomputes \(O(N)\) block work in total, and there are \(O(\log N)\) levels, so the total compute grows as \(O(N\log N)\).

### Clarification

An even more aggressive custom recomputation thought experiment could repeatedly recompute prefixes from the model input and keep only the current block's residuals, suggesting \(O(1)\) activation storage at the cost of \(O(N^2)\) recomputation. That is **not** the balanced nested-checkpoint construction used for this assignment answer.

## (b) XL Model Checkpoint Block-Size Profiling

**Skipped / not completed at the user's request.**

The assignment asks for profiling the XL model with batch size 4 and sequence length 2048 under a single level of recomputation, then comparing the best checkpoint block size with the next smaller and larger choices. This experiment was intentionally skipped, so no measured peak-memory values are fabricated here.
