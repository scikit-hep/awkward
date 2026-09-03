# GPU backend and cuDF compatibility

## Supported cuDF versions

The GPU backend is primarily tested against the cuDF versions used in CI.
Local installations may differ due to ABI or constructor changes between
cuDF releases.

## Known limitations

- Conversion of deeply nested or jagged Awkward Arrays may fail on some cuDF versions.
- Mask handling behavior differs across cuDF releases.
- Some issues may only reproduce in CI environments.

## CI vs local environment differences

CI uses pinned RAPIDS images, while local environments may expose differences
in cuDF constructor signatures or column initialization behavior.

## Common error symptoms

- `TypeError` raised during cuDF column construction
- Mask silently dropped or ignored
- Conversion failures that do not reproduce locally
