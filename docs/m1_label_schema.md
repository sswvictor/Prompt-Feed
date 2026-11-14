# M1 Label and BIO Schema

## Intent labels (`service_category`)

- `fitness`
- `wellness`
- `beauty`
- `medical`
- `dining`
- `travel`
- `other`
- `ambiguous` (deprecated for auto-gate; retained for backward compat when explicitly predicted; vague inputs handled by M2 multi-turn dialogue)

## Slot fields (STV)

- `temporal_preference`: `ASAP | weekend | morning | afternoon | evening | tomorrow | null`
- `urgency_level`: `low | medium | high`
- `location_context`: `nearby | city_center | null`
- `sentiment_tags`: free list from controlled vocabulary (`relaxing`, `affordable`, `quick`, `premium`, `gentle`, `intense`)

## BIO tags for slot evaluation

- `O`
- `B-TEMPORAL`, `I-TEMPORAL`
- `B-URGENCY`, `I-URGENCY`
- `B-LOCATION`, `I-LOCATION`

## Split policy

- Seed fixed to 42
- Ratios: train 0.70 / val 0.15 / test 0.15
- Manifest file: `data/annotations/m1_split_manifest.json`
