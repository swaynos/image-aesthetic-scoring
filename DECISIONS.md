# Decisions

## 3.0.0 — Remove ImageReward

**Date:** 2026-09-12

**Decision:** Drop the ImageReward scorer from the suite entirely. Remove the
`score_imagereward` function, the `ImageRewardScoreResult` type, the
`imagereward` model identifier, the `[imagereward]` optional dependency, and
the unit and smoke tests. This is a breaking change, so the package version
moves from 2.0.0 to 3.0.0.

**Why.**

ImageReward cannot run in this project's environment, and the reason is not
fixable on our side without breaking two scorers that work.

- The published package `image-reward` 1.5 is the only release on PyPI and
  dates from July 2023. There has been no release since.
- It vendors its own copy of BLIP's BERT, written against transformers
  *internal* APIs. Those internals changed in transformers 5.x. Under the
  transformers 5.17 this project pins, `import ImageReward` fails outright —
  it imports `apply_chunking_to_forward` from `transformers.modeling_utils`
  and uses `transformers.file_utils`, both gone in 5.x.
- Upstream (now `zai-org/ImageReward`) merged one partial compatibility fix
  (PR #118, October 2025) but never cut a release, so it never reached PyPI.
  A fuller "transformers >= 5" fix (PR #123) remains open and unmerged, from
  an outside contributor, tested only against transformers 5.14.1.
- The only route to a working ImageReward is downgrading transformers to 4.x.
  PickScore and HPSv2 both depend on transformers and both pass today. Trading
  two working scorers for one that needs an unmerged third-party patch is a bad
  deal for a scoring library, where a wrong number is worse than a missing one.

**Alternatives rejected.**

- *Keep it as an optional, skipped extra.* Leaves a dependency in the tree that
  can never be satisfied against the pinned transformers, and a default
  `score_images(...)` still could not include it. Dead weight that invites the
  same investigation again.
- *Pin transformers to 4.x.* High blast radius onto PickScore and HPSv2.
- *Install from a fork or PR #123.* Unmerged, third-party, tested on a
  different transformers version than we run, with no reference values to catch
  silently altered rewards.

**Revisit if:** upstream cuts a PyPI release that imports cleanly under
transformers 5.x. At that point the scorer can return as a core feature.

**Reference:** zai-org/ImageReward PRs #118, #123, issue #122.
