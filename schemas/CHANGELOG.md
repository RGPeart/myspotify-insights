# Schema Changelog

All notable changes to the pipeline dataset schemas are recorded here. Versions
follow semantic versioning. The Pydantic models in `src/schemas/` are the source
of truth; the JSON Schema files in this directory are generated from them via
`python scripts/generate_schemas.py`. Bump the version in
`src/schemas/registry.py`, regenerate, and add an entry below on every change.

## [1.0.0] - 2026-06-09

### Initial schema definitions for the silver and gold layers

Captures the contracts as produced by the current pipeline:

**Silver** (from `src/etl/bronze_to_silver.py`):
- `silver/tracks` — track_id, name, popularity, duration_ms, explicit, primary_artist_id, primary_artist_name, album_id, album_name, release_date
- `silver/audio_features` — track_id + normalized audio features (danceability, energy, tempo required; key, loudness, time_signature, speechiness, acousticness, instrumentalness, liveness, valence, mode, duration_ms optional)
- `silver/artists` — artist_id, name, popularity, followers, genres, primary_genre

**Gold** (from `dbt/models/gold/`):
- `gold/dim_tracks` — track dimension with composite_popularity (0–1) and primary_genre
- `gold/dim_artists` — artist dimension (popularity renamed from artist_popularity)
- `gold/fact_audio_features` — audio-feature fact joined to track/artist/album keys

Bronze is intentionally excluded: it stores raw, nested Spotify API JSON whose
shape is controlled by Spotify, not this pipeline.
