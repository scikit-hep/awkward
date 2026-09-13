# Awkward Array documentation


% TODO mention left side-bar
:::::{grid} 1 1 2 2
:class-container: intro-grid text-center

::::{grid-item-card} 
:columns: 12

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

:::{image} https://img.shields.io/conda/vn/conda-forge/awkward?style=for-the-badge&color=brightgreen&logo=condaforge
:alt: Conda (channel only)
:target: https://anaconda.org/conda-forge/awkward
:class: shield-badge
:::

:::{image} https://img.shields.io/pypi/v/awkward?style=for-the-badge&logo=pypi
:alt: PyPI
:target: https://pypi.org/project/awkward
:class: shield-badge
:::

:::{image} https://img.shields.io/badge/GitHub-scikit--hep%2Fawkward-lightgrey?style=for-the-badge&logo=GitHub
:alt: GitHub
:target: https://github.com/scikit-hep/awkward
:class: shield-badge
:::

:::{image} https://img.shields.io/badge/-Try%20It%21%20%E2%86%97-orange?style=for-the-badge
:alt: Try It! ⭷
:target: _static/try-it.html
:class: shield-badge
:::

::::

::::{grid-item-card} 
:link-type: doc
:link: getting-started/index

{fas}`running`

Getting started 
^^^^^^^^^^^^^^^
New to *Awkward Array*? Unsure what it can be used for? Check out the getting started guides. They contain an introduction to *Awkward Array's* features and links to additional help.
   

::::

:::{grid-item-card}
:link-type: doc
:link: user-guide/index

{fas}`book-open`

User guide
^^^^^^^^^^

The user guide provides in-depth information on the key concepts of Awkward Array with useful background information and explanation.

:::

:::{grid-item-card}
:link-type: doc
:link: reference/index

{fas}`code`

API reference
^^^^^^^^^^^^^

The reference guide contains a detailed description of the functions, modules, and objects included in Awkward Array. The reference describes how the methods work and which parameters can be used. It assumes that you have an understanding of the key concepts.

:::

:::{grid-item-card}
:link: https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md

{fas}`terminal`

Contributor's guide
^^^^^^^^^^^^^^^^^^^
Spotted a typo in the documentation? Want to add to the codebase? The contributing guidelines will guide you through the process of improving Awkward Array.

:::

:::{grid-item-card} 
:columns: 12
:link: https://awkward-array.org/doc/1.10/api-reference.html
:class-card: admonition warning

{fas}`history` Need the documentation for version 1 of Awkward Array? Click this card.

:::

:::{grid-item-card} 
:columns: 12
:link: https://dask-awkward.readthedocs.io/
:class-card: admonition warning

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNC4zNzYyMzk4bW0iIGhlaWdodD0iNC4wNzkxODE3bW0iIHZpZXdCb3g9IjAgMCA0LjM3NjIzOTggNC4wNzkxODE3IiB2ZXJzaW9uPSIxLjEiIHN0eWxlPSJmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MC4wMDA0ODU0NDQ7IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPiA8cGF0aCBkPSJNIDEuMjg4NjQ5NCwxLjIyNzIyMDQgMi4zNDQxMDY2LDAuNjE4MjI1NzQgYyAwLjAxMDQyOCwtMC4wMDYwMDkgMC4wMTY4NTIsLTAuMDE3MTAzOCAwLjAxNjg1MiwtMC4wMjkyMDA4IGwgOC4xOWUtNSwtMC4zNjUxOTY2MiBjIDAsLTAuMDU0MDY2MSAtMC4wMjIxOTQsLTAuMTA3NjMxNTkgLTAuMDY1MzMsLTAuMTQwMjU0NzUgQyAyLjIzOTg5MjYsMC4wNDEzNTcyNiAyLjE2ODIyMTQsMC4wMzcxODQ5NiAyLjEwOTM5OTQsMC4wNzEyMjQ2IEwgMC42MzMwMTExMSwwLjkyMzE4Mjc5IGMgLTAuMDUxODk5MiwwLjAyOTk1MTMgLTAuMDgzOTM2MSwwLjA4NTM1NDYgLTAuMDgzOTM2MSwwLjE0NTI2MTAxIGwgLTUuNzczZS00LDEuOTI0MDE5NCBjIDAsMC4wNTM5ODMgMC4wMjIxMTA1LDAuMTA3NTQ4MiAwLjA2NTE2MjksMC4xNDAyNTQ3IDAuMDU1NzM1MSwwLjA0MjMwMSAwLjEyNzU3MjU0LDAuMDQ2NDcyIDAuMTg2NDc4MDIsMC4wMTI0MzMgTCAxLjExMjg1NDcsMi45NjQ3NjM0IGMgMC4wMTA0MjgsLTAuMDA2MDEgMC4wMTY4NTIsLTAuMDE3MTA0IDAuMDE2ODUyLC0wLjAyOTIwMSBsIDQuMTcyZS00LC0xLjQzMzgzNjYgYyAwLC0wLjExMzMwNTQgMC4wNjA0MDcsLTAuMjE3ODUgMC4xNTg1MjY4LC0wLjI3NDUwMjQgeiIgc3R5bGU9ImZpbGw6I2ZmYzExZTsiIC8+IDxwYXRoIGQ9Ik0gMy44MTg3NDMsMC45NTEyMTY0MiBDIDMuNzkyNDYsMC45MzYwMjgxMiAzLjc2MzU5MiwwLjkyODQzOTMyIDMuNzM0ODA4LDAuOTI4NDM5MzIgYyAtMC4wMjg3ODYsMCAtMC4wNTc1NzEsMC4wMDc1OTQgLTAuMDgzODUzLDAuMDIyNzc3MSBMIDIuMTc0NDgyNSwxLjgwMzA5MTEgYyAtMC4wNTE3MzEsMC4wMjk4NzMgLTAuMDgzOTM2LDAuMDg1NTIxIC0wLjA4MzkzNiwwLjE0NTI2MTEgbCAtNS43NzRlLTQsMS45MzEyNzc2IGMgMCwwLjA2MDY1OCAwLjAzMTM3MywwLjExNDk3NDMgMC4wODM4NTIsMC4xNDUzNDQ0IDAuMDUyNTY1LDAuMDMwMzcxIDAuMTE1MjI0NywwLjAzMDM3MSAwLjE2Nzc4ODgsMCBMIDMuODE3OTk4MiwzLjE3MzAxNiBjIDAuMDUxNzMxLC0wLjAyOTg3MyAwLjA4MzkzNiwtMC4wODU1MjEgMC4wODM5MzYsLTAuMTQ1MjYwOSBsIDYuODIyZS00LC0xLjkzMTE5NDggYyAwLC0wLjA2MDY1OCAtMC4wMzEzNzMsLTAuMTE1MDU3MjQgLTAuMDgzODUzLC0wLjE0NTM0Mzg4IHoiIHN0eWxlPSJmaWxsOiNlZjExNjE7IiAvPiA8cGF0aCBkPSJNIDIuMDk5OTc0NywxLjY3Mzg0OTcgMy4wNzQzMzI5LDEuMTExNjYyNCBjIDAuMDEwNDI4LC0wLjAwNjAxIDAuMDE2ODUyLC0wLjAxNzEwNCAwLjAxNjg1MiwtMC4wMjkyMDEgbCAxLjU3NGUtNCwtMC40MjUzNTMyMiBjIDAsLTAuMDU0MDY2MSAtMC4wMjIxOTQsLTAuMTA3NjMxNTggLTAuMDY1MzMsLTAuMTQwMjU1MjcgQyAyLjk3MDE5NDIsMC40NzQ2MzY4IDIuODk4NjA2OCwwLjQ3MDU0ODQ3IDIuODM5NzAxMywwLjUwNDUwNDE2IEwgMi40MzU2MzM2LDAuNzM3NzA2MTMgMS4zNjMxNTU2LDEuMzU2NDYyOSBjIC0wLjA1MTg5OSwwLjAyOTk1MSAtMC4wODM5MzYsMC4wODUzNTUgLTAuMDgzOTM2LDAuMTQ1MjYwOSBsIC00LjE3MmUtNCwxLjQ1MzI3NjkgLTEuNjc0ZS00LDAuNDcwNjU4NiBjIDAsMC4wNTQwNjYgMC4wMjIxMSwwLjEwNzU0ODIgMC4wNjUxNjMsMC4xNDAyNTQ3IDAuMDU1ODE4LDAuMDQyMzAxIDAuMTI3NTcyNSwwLjA0NjQ3MiAwLjE4NjQ3OCwwLjAxMjQzMyBMIDEuOTI0MDkwNywzLjM1MTA2OTIgYyAwLjAxMDQyOCwtMC4wMDYwMSAwLjAxNjg1MiwtMC4wMTcxMDQgMC4wMTY4NTIsLTAuMDI5MjAxIEwgMS45NDEzNiwxLjk0ODE4ODggQyAxLjk0MTQ0MTUsMS44MzUwNTAzIDIuMDAxODUsMS43MzA0MjIzIDIuMDk5OTcwNCwxLjY3Mzg1MzMgWiIgc3R5bGU9ImZpbGw6I2ZjNmU2YjsiIC8+IDwvc3ZnPg=="> Using dask-awkward arrays in Dask? Click this card.
::::

:::{grid-item-card} 
:columns: 12
:link: https://juliahep.github.io/AwkwardArray.jl/dev/
:class-card: admonition warning

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNC4zNzYyMzk4bW0iIGhlaWdodD0iNC4wNzkxODE3bW0iIHZpZXdCb3g9IjAgMCA0LjM3NjIzOTggNC4wNzkxODE3IiB2ZXJzaW9uPSIxLjEiIHN0eWxlPSJmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MC4wNTgxODI7IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPiA8Y2lyY2xlIHN0eWxlPSJmaWxsOiNjYjNjMzM7IiBjeD0iMS4wNzA5NSIgY3k9IjMuMDA4MjMxIiByPSIwLjk3MDk1MTU2IiAvPiA8Y2lyY2xlIHN0eWxlPSJmaWxsOiM5NTU4YjI7IiBjeD0iMy4zMDUyOSIgY3k9IjMuMDA4MjMxIiByPSIwLjk3MDk1MTU2IiAvPiA8Y2lyY2xlIHN0eWxlPSJmaWxsOiMzODk4MjY7IiBjeD0iMi4xODgxNCIgY3k9IjEuMDcwOTUyIiByPSIwLjk3MDk1MTU2IiAvPiA8L3N2Zz4="> Using AwkwardArray.jl in Julia? Click this card.
::::

:::::
