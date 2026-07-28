# Repository safety rules

1. Never mutate source while E00 or E01 is active.
2. Preserve existing E00/E01 artifacts as `v6.0-pilot`; do not relabel them as v6.1 confirmatory evidence.
3. Apply the v6.1 patch only after the active-process check is empty.
4. Never overwrite artifacts; every run gets a new UTC directory.
5. Do not silently substitute reference/mock code for an official backend.
6. Do not change metrics, SESOI, equivalence margins, or claim wording after test evaluation.
7. Do not launch long GPU jobs or model downloads without explicit instruction.
8. E06/E07 remain secondary unless official quality-controlled evidence exists.
