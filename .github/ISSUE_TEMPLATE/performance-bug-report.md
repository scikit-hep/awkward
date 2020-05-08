---
name: Performance bug report
about: It works, but it could/should be faster…
title: ''
labels: performance
assignees: ''

---

The goal of these issues is to fix performance "mistakes," instances where a fix would make the majority of applications several times faster or more, not fine-tuning an application or trading performance in one case for another (unless the former is a very rare or unusual case).

To prove that something is a performance mistake, it needs to have a reproducible metric and a demonstration that shows how fast it could be, such as equivalent C or Numba code. If the comparison is truly equivalent (i.e. a general-purpose function is not compared with a highly specialized one), we'll try to optimize the metric within a factor of 2 or so of the baseline.

Alternatively, if you've found a mistake in the code that would always be faster if fixed, we can fix it without tests. Some bugs are obvious.
