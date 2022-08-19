# Try it in your browser

Awkward Array even runs in the browser using JupyterLite! Use the console below to try out some of Awkward Array's features.

:::{replite}
   :kernel: python
   :height: 600px

   # Install Awkward Array in the browser
   import piplite
   await piplite.install("awkward")

   # Import Awkward Array as normal
   import awkward as ak
   x = ak.Array([
       [1, 2, 3],
       [4, 5],
       [6, 7, 8, 9]
   ])
   ak.sum(x, axis=-1)
:::
