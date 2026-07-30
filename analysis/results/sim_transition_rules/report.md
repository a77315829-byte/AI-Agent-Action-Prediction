# SIM Transition Rule Analysis

## Qwen baseline

- Validation Macro-F1: `0.779135`
- Pooled best rule: `0.779844`
- Nested rule estimate: `0.779310`
- Fixed chain rule: `0.779557`

## Main finding

The synthetic `sim` sessions contain a strong two-turn exploration chain:

| Action at t-2 | Action at t | Rate |
|---|---|---:|
| `list_directory` | `read_file` | 64.87% |
| `read_file` | `grep_search` | 84.95% |
| `grep_search` | `glob_pattern` | 81.85% |

Qwen already follows the `read_file -> grep_search` and
`grep_search -> glob_pattern` transitions in most validation
samples. The remaining actionable rule is:

```text
second_last_action == list_directory
AND source == sim
AND Qwen predicts an Explore action other than read_file
=> override to read_file
```

This changes `28` validation samples and moves
Macro-F1 from `0.779135` to `0.779557`.

## Interpretation

- `au` Explore performance is already high.
- `sim` Explore performance is the main bottleneck.
- One-step transition state is weak.
- Two-turn state is substantially more predictive.
- Generic threshold/grid rules overfit the single validation fold.
- The fixed exploration-chain rule is interpretable and more stable,
  but its gain is still small.
