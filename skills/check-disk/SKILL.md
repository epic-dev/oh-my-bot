---
name: check-disk
description: Investigate disk usage and find what is taking up space.
---

To investigate disk usage:

1. Run `df -h` to see which filesystem is full.
2. Run `du -sh */ | sort -rh | head -20` from the relevant directory to find the
   largest subdirectories.
3. Descend into the largest one and repeat until you find the cause.

Report the top few offenders with their sizes. Do not delete anything.
