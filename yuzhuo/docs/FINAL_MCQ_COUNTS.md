# Final MCQ Counts

This file summarizes the counts for the final question sets actually used per species.

## Selection Rule

The provided `mcq_counts` match the following protocol selections:

* `P1`: `N4`
* `P2`: `N4_K1`
* `P3`: `N4_M2`
* `P4`: `N4`
* `P5`: `grayscale_S1_N4`, `occlusion_S1_N4`, `resolution_S1_N4`
* `P6`: `N4`
* `P7`: full protocol file

Notes:
* For the `MetaWild` species (`Stoat`, `Hare`, `Wallaby`, `Deer`, `Penguin`, `Pukeko`), the final count comes directly from `P4 N4`.
* For `P5`, `grayscale` only has `S1`, while `occlusion` and `resolution` have `S1/S2/S3`; at `N4`, their task counts are the same across severities in this repository, so using `S1_N4` reproduces the supplied totals exactly.

## Species Summary

| Species | P1 N4 | P2 N4 K1 | P3 N4 M2 | P4 N4 | P5 Grayscale N4 | P5 Occlusion N4 | P5 Resolution N4 | P6 N4 | P7 | Computed Total MCQs | Provided MCQ Count | Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BelugaID | 2351 | 1532 | 2069 |  | 2351 | 2351 | 2351 | 2657 | 3940 | 19602 | 19602 | Yes |
| SealID | 2080 | 2080 | 2080 |  | 2080 | 2080 | 2080 | 2080 | 284 | 14844 | 14844 | Yes |
| NDD20 | 2657 | 2657 | 2657 |  | 2657 | 2657 | 2657 | 2657 | 409 | 19008 | 19008 | Yes |
| NyalaData | 1941 | 1662 | 1829 |  | 1941 | 1941 | 1941 | 1942 | 1185 | 14382 | 14382 | Yes |
| IPanda50 | 4975 | 5455 | 6874 |  | 4975 | 4975 | 4975 | 6874 | 250 | 39353 | 39353 | Yes |
| BirdIndividualID | 957 | 785 | 1275 |  | 957 | 957 | 957 | 1275 | 128 | 7291 | 7291 | Yes |
| Giraffes | 1368 | 1362 | 1366 |  | 1368 | 1368 | 1368 | 1368 | 890 | 10458 | 10458 | Yes |
| LeopardID2022 | 6756 | 6227 | 6592 |  | 6756 | 6756 | 6756 | 6806 | 2150 | 48799 | 48799 | Yes |
| HumpbackWhaleID | 13624 | 8258 | 11054 |  | 13624 | 13624 | 13624 | 15697 | 25018 | 114523 | 114523 | Yes |
| Stoat |  |  |  | 6733 |  |  |  |  |  | 6733 | 6733 | Yes |
| Hare |  |  |  | 2860 |  |  |  |  |  | 2860 | 2860 | Yes |
| Wallaby |  |  |  | 1937 |  |  |  |  |  | 1937 | 1937 | Yes |
| Deer |  |  |  | 1764 |  |  |  |  |  | 1764 | 1764 | Yes |
| Penguin |  |  |  | 2452 |  |  |  |  |  | 2452 | 2452 | Yes |
| Pukeko |  |  |  | 972 |  |  |  |  |  | 972 | 972 | Yes |
| CTai | 4661 | 4617 | 4657 |  | 4661 | 4661 | 4661 | 4662 | 352 | 32932 | 32932 | Yes |
| Lion | 740 | 726 | 740 |  | 740 | 740 | 740 | 740 | 469 | 5635 | 5635 | Yes |
| WhaleSharkID | 7662 | 7171 | 7572 |  | 7662 | 7662 | 7662 | 7693 | 2715 | 55799 | 55799 | Yes |

## Overall Check

* `All species matched`: `Yes`
* `Total computed MCQs`: `399344`
* `Total provided MCQs`: `399344`
