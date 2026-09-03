---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to find the best match between two collections using Cartesian (cross) product
==================================================================================

In high energy physics (HEP), {func}`ak.combinations` is often needed to find particles whose trajectories are close to each other, separately in many high-energy collision events (`axis=1`). In some applications, the two collections that need to be matched are simulated particles and reconstructed versions of those particles ("gen-reco matching"), and in other applications, the two collections are different types of particles, such as electrons and jets.

I'll describe how to solve such a problem on this page, but avoid domain-specific jargon by casting it as a problem of finding the distance between bunnies and foxes—if a bunny is too close to a fox, it will get eaten!

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## Setting up the problem

In 1000 separate yards (big suburb), there's a random number of bunnies and a random number of foxes, each with random _x_, _y_ positions. We're making ragged arrays of records using {func}`ak.unflatten` and {func}`ak.zip`.

```{code-cell} ipython3
np.random.seed(12345)

number_of_bunnies = np.random.poisson(3.5, 1000)   # average of 3.5 bunnies/yard
number_of_foxes = np.random.poisson(1.5, 1000)     # average of 1.5 foxes/yard

bunny_xy = np.random.normal(0, 1, (number_of_bunnies.sum(), 2))
fox_xy = np.random.normal(0, 1, (number_of_foxes.sum(), 2))

bunnies = ak.unflatten(ak.zip({"x": bunny_xy[:, 0], "y": bunny_xy[:, 1]}), number_of_bunnies)
foxes = ak.unflatten(ak.zip({"x": fox_xy[:, 0], "y": fox_xy[:, 1]}), number_of_foxes)
```

```{code-cell} ipython3
bunnies
```

```{code-cell} ipython3
foxes
```

## Find all combinations

In each yard, we find all bunny-fox pairs, regardless of whether they're close or not using {func}`ak.cartesian`, and then unpacking the pairs with {func}`ak.unzip`.

```{code-cell} ipython3
pair_bunnies, pair_foxes = ak.unzip(ak.cartesian([bunnies, foxes]))
```

These two arrays, `pair_bunnies` and `pair_foxes`, have the same type as `bunnies` and `foxes`, but different numbers of items in each list because now they're paired to match each other. Both kinds of animals are duplicated to enable this match.

```{code-cell} ipython3
pair_bunnies
```

```{code-cell} ipython3
pair_foxes
```

The two arrays have the same list lengths as each other because they came from the same {func}`ak.unzip`.

```{code-cell} ipython3
ak.num(pair_bunnies), ak.num(pair_foxes)
```

## Calculating distances

Since the arrays have the same shapes, they can be used in the same mathematical formula. Here's the formula for distance:

```{code-cell} ipython3
distances = np.sqrt((pair_bunnies.x - pair_foxes.x)**2 + (pair_bunnies.y - pair_foxes.y)**2)
distances
```

Let's say that 1 unit is close enough for a bunny to be eaten.

```{code-cell} ipython3
eaten = (distances < 1)
eaten
```

This is great (not for the bunnies, but perhaps for the foxes). However, if we want to use this information on the original arrays, we're stuck: this array has a different shape from the original `bunnies` (and the original `foxes`).

Perhaps the question we really wanted to ask is, "For each bunny, is there _any_ fox that can eat it?"

## Combinations with `nested=True`

Asking a question about _any_ fox means performing a reducer, {func}`ak.any`, over lists, one list per bunny. The list would be all of the foxes in its yard. For that, we'll need to pass `nested=True` to {func}`ak.cartesian`.

```{code-cell} ipython3
pair_bunnies, pair_foxes = ak.unzip(ak.cartesian([bunnies, foxes], nested=True))
```

Now `pair_bunnies` and `pair_foxes` are one list-depth deeper than the original `bunnies` and `foxes`.

```{code-cell} ipython3
pair_bunnies
```

```{code-cell} ipython3
pair_foxes
```

We can compute `distances` in the same way, though it's also one list-depth deeper.

```{code-cell} ipython3
distances = np.sqrt((pair_bunnies.x - pair_foxes.x)**2 + (pair_bunnies.y - pair_foxes.y)**2)
distances
```

Similarly for `eaten`.

```{code-cell} ipython3
eaten = (distances < 1)
eaten
```

Now each inner list of booleans is answering the questions, "Can fox 0 eat me?", "Can fox 1 eat me?", ..., "Can fox _n_ eat me?" and there are exactly as many of these lists as there are bunnies. Applying {func}`ak.any` over the innermost lists (`axis=-1`),

```{code-cell} ipython3
bunny_eaten = ak.any(eaten, axis=-1)
bunny_eaten
```

We've now answered the question, "Can any fox eat me?" for each bunny. After the mayhem, these are the bunnies we have left:

```{code-cell} ipython3
bunnies[~bunny_eaten]
```

Whereas there was originally an average of 3.5 bunnies per yard, by construction,

```{code-cell} ipython3
ak.mean(ak.num(bunnies, axis=1))
```

Now there's only

```{code-cell} ipython3
ak.mean(ak.num(bunnies[~bunny_eaten], axis=1))
```

left.

## Asymmetry in the problem

The way we performed this calculation was asymmetric: for each bunny, we asked if it was eaten. We could have performed a similar, but different, calculation to ask, which foxes get to eat? To do that, we must reverse the order of arguments because `nested=True` groups from the left.

```{code-cell} ipython3
pair_foxes, pair_bunnies = ak.unzip(ak.cartesian([foxes, bunnies], nested=True))

distances = np.sqrt((pair_foxes.x - pair_bunnies.x)**2 + (pair_foxes.y - pair_bunnies.y)**2)

eating = (distances < 1)

fox_eats = ak.any(eating, axis=-1)

foxes[fox_eats]
```
