# How to examine a single item in detail

It’s often useful to pull out a single item from an array to inspect its contents, particularly in the early stages of a data analysis, to get a sense of the data’s structure. This tutorial shows how to extract one item from an Awkward Array and examine it in different ways.

For this example, we’ll to use the Chicago taxi trips dataset from [10 minutes to Awkward Array](https://awkward-array.org/doc/main/getting-started/10-minutes-to-awkward-array.html). Recall that this dataset includes information about trips by various taxis collected over a few years, enriched with GPS path data.

## Loading the dataset

First, let’s load the dataset using the [`ak.from_parquet()`](sphinx-llm:ab272f6edef4436f8edba845c0477c5f#ak.from_parquet) function. We will only load the first row group, for the sake of this demonstration:

```ipython3
import awkward as ak

url = "https://zenodo.org/records/14537442/files/chicago-taxi.parquet"
taxi = ak.from_parquet(
    url,
    row_groups=[0],
    columns=["trip.km", "trip.begin.l*", "trip.end.l*", "trip.path.*"],
)
```

## What is a single item?

The first “item” of this dataset could be a single taxi, which comprises many trips.

```ipython3
single_taxi = taxi[5]
single_taxi
```

Or it could be a single trip.

```ipython3
single_trip = single_taxi.trip[5]
single_trip
```

Or it could be a single latitude, longitude position along the path.

```ipython3
single_trip.path
```

```ipython3
single_point = single_trip.path[5]
single_point
```

```ipython3
print(f"longitude: {single_trip.begin.lon + single_point.londiff:.3f}")
print(f"latitude:  {single_trip.begin.lat + single_point.latdiff:.3f}")
```

```myst-ansi
longitude: -87.899
latitude:  41.981
```

In Jupyter notebooks (and this documentation), the array contents are presented in a multi-line format with the data type below a dashed line.

## Standard Python `repr`

In a Python prompt, the format is more concise:

```ipython3
print(f"{single_taxi!r}")
```

```myst-ansi
<Array [{trip: {km: 29.6, ...}}, ..., {...}] type='2403 * ?{trip: {km: ?flo...'>
```

```ipython3
print(f"{single_trip!r}")
```

```myst-ansi
<Record {km: 18.6, begin: {...}, end: ..., ...} type='{km: ?float32, begin:...'>
```

```ipython3
print(f"{single_point!r}")
```

```myst-ansi
<Record {londiff: 0.0146, latdiff: ..., ...} type='{londiff: float32, latdi...'>
```

The long form can be obtained in a Python prompt with the `show` method:

```ipython3
single_taxi.show()
```

```myst-ansi
[{trip: {km: 29.6, begin: {lon: -87.9, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 2, begin: {lon: -87.6, ...}, end: {...}, path: [...]}},
 {trip: {km: 2.46, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 50.8, begin: {lon: -87.9, ...}, end: {...}, path: []}},
 {trip: {km: 27, begin: {lon: -87.8, ...}, end: {...}, path: [...]}},
 {trip: {km: 18.6, begin: {lon: -87.9, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 20.2, begin: {lon: -87.9, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 2.27, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 0.724, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 1.05, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 ...,
 {trip: {km: 2.77, begin: {lon: -87.7, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 1.37, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 1.74, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 0.612, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 14.4, begin: {lon: -87.6, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 24.1, begin: {lon: -87.9, ...}, end: {...}, path: []}},
 {trip: {km: 33.4, begin: {lon: -87.9, ...}, end: {...}, path: []}},
 {trip: {km: 24.3, begin: {lon: -87.9, ...}, end: {...}, path: ..., ...}},
 {trip: {km: 1.83, begin: {lon: -87.7, ...}, end: {...}, path: ..., ...}}]
```

```ipython3
single_trip.show()
```

```myst-ansi
{km: 18.6,
 begin: {lon: -87.9, lat: 42},
 end: {lon: -87.7, lat: 42},
 path: [{londiff: 0.00768, latdiff: -0.001}, {...}, ..., {londiff: 0.19, ...}]}
```

```ipython3
single_point.show()
```

```myst-ansi
{londiff: 0.0146,
 latdiff: 0.000311}
```

## The `show` method

The `show` method can take a `type=True` argument to include the type as well (at the top this time, because values are presented in the “most valuable real estate,” which is the bottom of a print-out in the terminal, but the top in a Jupyter notebook).

```ipython3
single_point.show(type=True)
```

```myst-ansi
type: {
    londiff: float32,
    latdiff: float32
}
{londiff: 0.0146,
 latdiff: 0.000311}
```

Types also have a `show` method, so if you *only* want the type, you can do

```ipython3
single_trip.type.show()
```

```myst-ansi
{
    km: ?float32,
    begin: {
        lon: ?float64,
        lat: ?float64
    },
    end: {
        lon: ?float64,
        lat: ?float64
    },
    path: var * {
        londiff: float32,
        latdiff: float32
    }
}
```

If you need to get this as a string or pass it to an output other than `sys.stdout`, use the `stream` parameter.

```ipython3
single_point.show(stream=None)
```

## Using `to_list` and Python’s `pprint` for a detailed view

The `repr` and `show` representations print into a restricted space: 1 line (80 characters) for `repr`, and 20 lines (80 character width) for `show` without `type=True`. To do this, they replace data with ellipses (`...`) until it fits.

You might want to ensure that you see everything. One way to do that is to turn the data into Python objects with [`ak.to_list()`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list) (or `to_list` or `tolist` as a method) and pretty-print them with Python’s `pprint`.

```ipython3
import pprint

trip_list = ak.to_list(single_trip)
pprint.pprint(trip_list)
```

```myst-ansi
{'begin': {'lat': 41.980264315, 'lon': -87.913624596},
 'end': {'lat': 41.953582125, 'lon': -87.72345239},
 'km': 18.60401725769043,
 'path': [{'latdiff': -0.00100231496617198, 'londiff': 0.007684595882892609},
          {'latdiff': -0.0011563149746507406, 'londiff': 0.007650596089661121},
          {'latdiff': -0.0015393149806186557, 'londiff': 0.00774059584364295},
          {'latdiff': -0.002198314992710948, 'londiff': 0.007572595961391926},
          {'latdiff': -0.0027953151147812605, 'londiff': 0.012805595993995667},
          {'latdiff': 0.0003106849908363074, 'londiff': 0.014558595605194569},
          {'latdiff': 0.0003526849905028939, 'londiff': 0.01837259531021118},
          {'latdiff': -0.002175315050408244, 'londiff': 0.02808559685945511},
          {'latdiff': 0.001839685020968318, 'londiff': 0.03775859624147415},
          {'latdiff': 0.002964684972539544, 'londiff': 0.04636159539222717},
          {'latdiff': 0.003171684918925166, 'londiff': 0.056837595999240875},
          {'latdiff': 0.0036336849443614483, 'londiff': 0.07005459815263748},
          {'latdiff': 0.004173684865236282, 'londiff': 0.0837985947728157},
          {'latdiff': 0.0029606849420815706, 'londiff': 0.09554759413003922},
          {'latdiff': 0.0019616850186139345, 'londiff': 0.1069725975394249},
          {'latdiff': 0.002038684906437993, 'londiff': 0.11698559671640396},
          {'latdiff': 0.0017906849971041083, 'londiff': 0.12975460290908813},
          {'latdiff': -0.005761315114796162, 'londiff': 0.1434255987405777},
          {'latdiff': -0.009354314766824245, 'londiff': 0.15160559117794037},
          {'latdiff': -0.013837315142154694, 'londiff': 0.16011258959770203},
          {'latdiff': -0.01785031519830227, 'londiff': 0.16788259148597717},
          {'latdiff': -0.023128315806388855, 'londiff': 0.17631658911705017},
          {'latdiff': -0.026048315688967705, 'londiff': 0.18113559484481812},
          {'latdiff': -0.026647314429283142, 'londiff': 0.18232059478759766},
          {'latdiff': -0.026622315868735313, 'londiff': 0.18967759609222412},
          {'latdiff': -0.026618314906954765, 'londiff': 0.19017159938812256}]}
```

Keep in mind that if you don’t slice a small enough section of data, your terminal or Jupyter notebook may be overwhelmed with output!

## Viewing data as JSON

Another way you can dump everything is to convert the data to JSON with [`ak.to_json()`](sphinx-llm:7af8124fc8d14a3caf95668155d7904d#ak.to_json).

```ipython3
print(ak.to_json(single_trip))
```

```myst-ansi
{"km":18.60401725769043,"begin":{"lon":-87.913624596,"lat":41.980264315},"end":{"lon":-87.72345239,"lat":41.953582125},"path":[{"londiff":0.007684595882892609,"latdiff":-0.00100231496617198},{"londiff":0.007650596089661121,"latdiff":-0.0011563149746507406},{"londiff":0.00774059584364295,"latdiff":-0.0015393149806186557},{"londiff":0.007572595961391926,"latdiff":-0.002198314992710948},{"londiff":0.012805595993995667,"latdiff":-0.0027953151147812605},{"londiff":0.014558595605194569,"latdiff":0.0003106849908363074},{"londiff":0.01837259531021118,"latdiff":0.0003526849905028939},{"londiff":0.02808559685945511,"latdiff":-0.002175315050408244},{"londiff":0.03775859624147415,"latdiff":0.001839685020968318},{"londiff":0.04636159539222717,"latdiff":0.002964684972539544},{"londiff":0.056837595999240875,"latdiff":0.003171684918925166},{"londiff":0.07005459815263748,"latdiff":0.0036336849443614483},{"londiff":0.0837985947728157,"latdiff":0.004173684865236282},{"londiff":0.09554759413003922,"latdiff":0.0029606849420815706},{"londiff":0.1069725975394249,"latdiff":0.0019616850186139345},{"londiff":0.11698559671640396,"latdiff":0.002038684906437993},{"londiff":0.12975460290908813,"latdiff":0.0017906849971041083},{"londiff":0.1434255987405777,"latdiff":-0.005761315114796162},{"londiff":0.15160559117794037,"latdiff":-0.009354314766824245},{"londiff":0.16011258959770203,"latdiff":-0.013837315142154694},{"londiff":0.16788259148597717,"latdiff":-0.01785031519830227},{"londiff":0.17631658911705017,"latdiff":-0.023128315806388855},{"londiff":0.18113559484481812,"latdiff":-0.026048315688967705},{"londiff":0.18232059478759766,"latdiff":-0.026647314429283142},{"londiff":0.18967759609222412,"latdiff":-0.026622315868735313},{"londiff":0.19017159938812256,"latdiff":-0.026618314906954765}]}
```

That’s not very readable, so we’ll pass `num_indent_spaces=4` to add newlines and indentation, and `num_readability_spaces=1` to add spaces after commas (`,`) and colons (`:`).

```ipython3
print(ak.to_json(single_trip, num_indent_spaces=4, num_readability_spaces=1))
```

```myst-ansi
{
    "km": 18.60401725769043, 
    "begin": {
        "lon": -87.913624596, 
        "lat": 41.980264315
    }, 
    "end": {
        "lon": -87.72345239, 
        "lat": 41.953582125
    }, 
    "path": [
        {
            "londiff": 0.007684595882892609, 
            "latdiff": -0.00100231496617198
        }, 
        {
            "londiff": 0.007650596089661121, 
            "latdiff": -0.0011563149746507406
        }, 
        {
            "londiff": 0.00774059584364295, 
            "latdiff": -0.0015393149806186557
        }, 
        {
            "londiff": 0.007572595961391926, 
            "latdiff": -0.002198314992710948
        }, 
        {
            "londiff": 0.012805595993995667, 
            "latdiff": -0.0027953151147812605
        }, 
        {
            "londiff": 0.014558595605194569, 
            "latdiff": 0.0003106849908363074
        }, 
        {
            "londiff": 0.01837259531021118, 
            "latdiff": 0.0003526849905028939
        }, 
        {
            "londiff": 0.02808559685945511, 
            "latdiff": -0.002175315050408244
        }, 
        {
            "londiff": 0.03775859624147415, 
            "latdiff": 0.001839685020968318
        }, 
        {
            "londiff": 0.04636159539222717, 
            "latdiff": 0.002964684972539544
        }, 
        {
            "londiff": 0.056837595999240875, 
            "latdiff": 0.003171684918925166
        }, 
        {
            "londiff": 0.07005459815263748, 
            "latdiff": 0.0036336849443614483
        }, 
        {
            "londiff": 0.0837985947728157, 
            "latdiff": 0.004173684865236282
        }, 
        {
            "londiff": 0.09554759413003922, 
            "latdiff": 0.0029606849420815706
        }, 
        {
            "londiff": 0.1069725975394249, 
            "latdiff": 0.0019616850186139345
        }, 
        {
            "londiff": 0.11698559671640396, 
            "latdiff": 0.002038684906437993
        }, 
        {
            "londiff": 0.12975460290908813, 
            "latdiff": 0.0017906849971041083
        }, 
        {
            "londiff": 0.1434255987405777, 
            "latdiff": -0.005761315114796162
        }, 
        {
            "londiff": 0.15160559117794037, 
            "latdiff": -0.009354314766824245
        }, 
        {
            "londiff": 0.16011258959770203, 
            "latdiff": -0.013837315142154694
        }, 
        {
            "londiff": 0.16788259148597717, 
            "latdiff": -0.01785031519830227
        }, 
        {
            "londiff": 0.17631658911705017, 
            "latdiff": -0.023128315806388855
        }, 
        {
            "londiff": 0.18113559484481812, 
            "latdiff": -0.026048315688967705
        }, 
        {
            "londiff": 0.18232059478759766, 
            "latdiff": -0.026647314429283142
        }, 
        {
            "londiff": 0.18967759609222412, 
            "latdiff": -0.026622315868735313
        }, 
        {
            "londiff": 0.19017159938812256, 
            "latdiff": -0.026618314906954765
        }
    ]
}
```

[`ak.to_json()`](sphinx-llm:7af8124fc8d14a3caf95668155d7904d#ak.to_json) is also one of the bulk output methods, so it can write data to a file, as a single JSON object or as `line_delimited` JSON.
