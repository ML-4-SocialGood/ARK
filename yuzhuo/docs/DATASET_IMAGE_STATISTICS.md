# Dataset Image Statistics

This document summarizes annotation coverage for each species and protocol under `annotations/`.

## Recommended Reading

For a dataset overview table, `Unique Images` should be treated as the primary scale indicator, and `Total Tasks` should be treated as a secondary workload indicator.

Why:
* `Unique Images` better reflects how much real visual content a protocol covers.
* `Total Tasks` is strongly affected by sampling strategy and protocol design, so it is useful, but it is not a pure dataset-size measure.
* This is especially important for `P5`, where multiple corruption variants can create many more tasks than the underlying clean-image pool would suggest.

## Counting Rule

* `Unique Images`: distinct `image_path` values aggregated across all JSON annotation files for one `species/protocol`.
* `Total Tasks`: total number of task entries across all JSON annotation files for one `species/protocol`.
* `Annotation Files`: number of JSON annotation files under that protocol directory.
* For `P5`, corrupted variants are counted as distinct images because they use different file paths.

## Table

| Species | Protocol | Unique Images | Total Tasks | Annotation Files |
|---|---|---:|---:|---:|
| BelugaID | p1 | 2657 | 9404 | 4 |
| BelugaID | p2 | 2657 | 24512 | 16 |
| BelugaID | p3 | 2657 | 20200 | 11 |
| BelugaID | p5 | 19114 | 65828 | 28 |
| BelugaID | p6 | 2657 | 10628 | 4 |
| BelugaID | p7 | 2175 | 3940 | 1 |
| BirdIndividualID | p1 | 1246 | 1040 | 4 |
| BirdIndividualID | p2 | 1182 | 3540 | 12 |
| BirdIndividualID | p3 | 1275 | 10200 | 11 |
| BirdIndividualID | p5 | 7947 | 7280 | 28 |
| BirdIndividualID | p6 | 1275 | 3825 | 4 |
| BirdIndividualID | p7 | 227 | 128 | 1 |
| CTai | p1 | 4662 | 9182 | 4 |
| CTai | p2 | 4640 | 35528 | 16 |
| CTai | p3 | 4662 | 51152 | 11 |
| CTai | p5 | 37271 | 64274 | 28 |
| CTai | p6 | 4662 | 18648 | 4 |
| CTai | p7 | 585 | 352 | 1 |
| Giraffes | p1 | 1368 | 5171 | 4 |
| Giraffes | p2 | 1368 | 21380 | 16 |
| Giraffes | p3 | 1368 | 15014 | 11 |
| Giraffes | p5 | 10944 | 36197 | 28 |
| Giraffes | p6 | 1368 | 5472 | 4 |
| Giraffes | p7 | 971 | 890 | 1 |
| HumpbackWhaleID | p1 | 15697 | 54496 | 4 |
| HumpbackWhaleID | p2 | 15695 | 132128 | 16 |
| HumpbackWhaleID | p3 | 15697 | 106390 | 11 |
| HumpbackWhaleID | p5 | 111065 | 381472 | 28 |
| HumpbackWhaleID | p6 | 15697 | 62788 | 4 |
| HumpbackWhaleID | p7 | 12140 | 25018 | 1 |
| IPanda50 | p1 | 6814 | 6757 | 4 |
| IPanda50 | p2 | 6750 | 27844 | 16 |
| IPanda50 | p3 | 6874 | 75614 | 11 |
| IPanda50 | p5 | 43749 | 47299 | 28 |
| IPanda50 | p6 | 6874 | 27496 | 4 |
| IPanda50 | p7 | 492 | 250 | 1 |
| LeopardID2022 | p1 | 6806 | 21246 | 4 |
| LeopardID2022 | p2 | 6760 | 79744 | 16 |
| LeopardID2022 | p3 | 6806 | 70757 | 11 |
| LeopardID2022 | p5 | 54058 | 148722 | 28 |
| LeopardID2022 | p6 | 6806 | 27224 | 4 |
| LeopardID2022 | p7 | 1899 | 2150 | 1 |
| Lion | p1 | 740 | 2576 | 4 |
| Lion | p2 | 740 | 11616 | 16 |
| Lion | p3 | 740 | 8074 | 11 |
| Lion | p5 | 5920 | 18032 | 28 |
| Lion | p6 | 740 | 2960 | 4 |
| Lion | p7 | 526 | 469 | 1 |
| MetaWild/Deer | p4 | 2263 | 2335 | 4 |
| MetaWild/Hare | p4 | 3050 | 5148 | 4 |
| MetaWild/Penguin | p4 | 2452 | 5003 | 4 |
| MetaWild/Pukeko | p4 | 1931 | 1138 | 4 |
| MetaWild/Stoat | p4 | 6733 | 18261 | 4 |
| MetaWild/Wallaby | p4 | 2635 | 3178 | 4 |
| NDD20 | p1 | 2657 | 6325 | 4 |
| NDD20 | p2 | 2656 | 24944 | 16 |
| NDD20 | p3 | 2657 | 29227 | 11 |
| NDD20 | p5 | 21256 | 44275 | 28 |
| NDD20 | p6 | 2657 | 10628 | 4 |
| NDD20 | p7 | 695 | 409 | 1 |
| NyalaData | p1 | 1942 | 7625 | 4 |
| NyalaData | p2 | 1942 | 25260 | 16 |
| NyalaData | p3 | 1942 | 19606 | 11 |
| NyalaData | p5 | 15529 | 53375 | 28 |
| NyalaData | p6 | 1942 | 7768 | 4 |
| NyalaData | p7 | 1155 | 1185 | 1 |
| SealID | p1 | 2080 | 5002 | 4 |
| SealID | p2 | 2080 | 18576 | 16 |
| SealID | p3 | 2080 | 22880 | 11 |
| SealID | p5 | 16640 | 35014 | 28 |
| SealID | p6 | 2080 | 8320 | 4 |
| SealID | p7 | 484 | 284 | 1 |
| WhaleSharkID | p1 | 7693 | 26907 | 4 |
| WhaleSharkID | p2 | 7691 | 97932 | 16 |
| WhaleSharkID | p3 | 7693 | 81525 | 11 |
| WhaleSharkID | p5 | 61327 | 188349 | 28 |
| WhaleSharkID | p6 | 7693 | 30772 | 4 |
| WhaleSharkID | p7 | 2988 | 2715 | 1 |
