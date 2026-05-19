# API Documentation

The MySpotify Insights REST API is a FastAPI service that serves music recommendations and metadata from the gold-layer data produced by the ETL pipeline.

Interactive Swagger UI is available at **`http://localhost:8001/docs`** once the server is running.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Starting the Server](#2-starting-the-server)
3. [Base URL & Versioning](#3-base-url--versioning)
4. [Prerequisites](#4-prerequisites)
5. [Endpoints](#5-endpoints)
   - [GET /health](#get-health)
   - [GET /users](#get-users)
   - [GET /recommendations/{user_id}](#get-recommendationsuser_id)
   - [GET /tracks/ids](#get-tracksids)
   - [GET /tracks/{track_id}](#get-trackstrack_id)
   - [GET /artists/{artist_id}](#get-artistsartist_id)
6. [Error Responses](#6-error-responses)
7. [Configuration](#7-configuration)

---

## 1. Overview

| Property | Value |
|---|---|
| Framework | FastAPI 0.109+ |
| Default port | `8001` |
| Auth | None (local development) |
| Response format | JSON |
| OpenAPI docs | `http://localhost:8001/docs` |
| ReDoc | `http://localhost:8001/redoc` |

At startup the server loads three gold-layer Parquet files (`dim_tracks`, `dim_artists`, `fact_audio_features`) and the trained recommendation model into memory. All reads are served from in-memory indexes — no database queries per request.

---

## 2. Starting the Server

```bash
# From the project root with your virtual environment active
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001
```

`--reload` watches for code changes and restarts automatically (development only — omit in production).

---

## 3. Base URL & Versioning

```
http://localhost:8001
```

There is currently no version prefix in the URL (e.g. `/v1/`). All endpoints are mounted at the root path.

---

## 4. Prerequisites

The following must be in place before the API returns non-503 responses:

| Requirement | How to satisfy |
|---|---|
| Gold-layer Parquet files exist | Run the `spotify_etl_pipeline` Airflow DAG (or `python -m src.etl.silver_to_gold` directly) |
| Trained recommendation model exists | Run `python -m src.models.train` after the gold layer is populated |

If either is missing the server still starts, but affected endpoints return `503 Service Unavailable` with a message explaining what needs to be run.

---

## 5. Endpoints

---

### GET /health

Returns the API health status and version. Always responds `200` regardless of whether the model or gold data is loaded.

**Request**

```
GET /health
```

**Response `200`**

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

**Example**

```bash
curl http://localhost:8001/health
```

---

### GET /users

Returns the list of user IDs known to the collaborative filtering model. These are the IDs you can pass to `/recommendations/{user_id}`.

**Request**

```
GET /users
```

**Response `200`**

```json
{
  "user_ids": ["user_abc123", "user_def456"]
}
```

**Response `503`** — model not loaded

```json
{
  "detail": "Collaborative model not loaded — run `python -m src.models.train` first"
}
```

**Example**

```bash
curl http://localhost:8001/users
```

---

### GET /recommendations/{user_id}

Returns personalised track recommendations for a given user using a hybrid collaborative + content-based model.

**Request**

```
GET /recommendations/{user_id}?n=10&liked_track_ids=<id1>,<id2>
```

**Path parameters**

| Parameter | Type | Description |
|---|---|---|
| `user_id` | string | User ID — must exist in the collaborative model (see `GET /users`) |

**Query parameters**

| Parameter | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `n` | integer | `10` | 1–100 | Number of recommendations to return |
| `liked_track_ids` | string | `null` | Comma-separated Spotify track IDs (22-char alphanumeric); max 5 used | Seed tracks for content-based signal; improves cold-start quality |

**Response `200`**

```json
{
  "user_id": "user_abc123",
  "count": 3,
  "recommendations": [
    {
      "track_id": "4uLU6hMCjMI75M1A2tKUQC",
      "score": 0.9231,
      "name": "Never Gonna Give You Up",
      "primary_artist_name": "Rick Astley",
      "primary_genre": "pop",
      "composite_popularity": 0.8714
    }
  ]
}
```

**Response fields**

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Echo of the requested user ID |
| `count` | integer | Number of recommendations returned |
| `recommendations[].track_id` | string | Spotify track ID |
| `recommendations[].score` | float | Hybrid recommendation score (higher = stronger match) |
| `recommendations[].name` | string | Track title |
| `recommendations[].primary_artist_name` | string | Lead artist name |
| `recommendations[].primary_genre` | string | Broad genre category |
| `recommendations[].composite_popularity` | float | Weighted popularity score 0–1 (60% track + 40% artist popularity) |

**Response `503`** — model not loaded

```json
{
  "detail": "Model not loaded — run `python -m src.models.train` first"
}
```

**Example**

```bash
# Basic — 10 recommendations for a user
curl "http://localhost:8001/recommendations/user_abc123"

# Request 5 recommendations seeded with two liked tracks
curl "http://localhost:8001/recommendations/user_abc123?n=5&liked_track_ids=4uLU6hMCjMI75M1A2tKUQC,7ouMYWpwJ422jRcDASZB7P"
```

---

### GET /tracks/ids

Returns the full list of track IDs available in the gold layer. Useful for populating dropdowns or selecting a random track for the dashboard.

**Request**

```
GET /tracks/ids
```

**Response `200`**

```json
{
  "track_ids": [
    "4uLU6hMCjMI75M1A2tKUQC",
    "7ouMYWpwJ422jRcDASZB7P"
  ]
}
```

**Response `503`** — gold data not loaded

```json
{
  "detail": "Track data not loaded"
}
```

**Example**

```bash
curl http://localhost:8001/tracks/ids
```

---

### GET /tracks/{track_id}

Returns full metadata and audio features for a single track.

**Request**

```
GET /tracks/{track_id}
```

**Path parameters**

| Parameter | Type | Description |
|---|---|---|
| `track_id` | string | Spotify track ID (22-character alphanumeric string) |

**Response `200`**

```json
{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "name": "Never Gonna Give You Up",
  "artist_id": "0gxyHStUsqpMadRV0Di1Qt",
  "primary_artist_name": "Rick Astley",
  "primary_genre": "pop",
  "composite_popularity": 0.8714,
  "audio_features": {
    "danceability": 0.614,
    "energy": 0.893,
    "valence": 0.898,
    "tempo": 0.453,
    "acousticness": 0.002,
    "instrumentalness": 0.0,
    "liveness": 0.096,
    "speechiness": 0.028,
    "loudness": 0.731,
    "mode": 1,
    "key": 0.417,
    "time_signature": 0.75
  }
}
```

> **Note:** All audio feature values are normalised to the range `[0, 1]` by the ETL pipeline. Raw Spotify values (e.g. tempo in BPM) are not exposed here.

**Response `404`** — track not found

```json
{
  "detail": "Track '4uLU6hMCjMI75M1A2tKUQC' not found"
}
```

**Response `503`** — gold data not loaded

```json
{
  "detail": "Track data not loaded"
}
```

**Example**

```bash
curl http://localhost:8001/tracks/4uLU6hMCjMI75M1A2tKUQC
```

---

### GET /artists/{artist_id}

Returns metadata for a single artist.

**Request**

```
GET /artists/{artist_id}
```

**Path parameters**

| Parameter | Type | Description |
|---|---|---|
| `artist_id` | string | Spotify artist ID (22-character alphanumeric string) |

**Response `200`**

```json
{
  "artist_id": "0gxyHStUsqpMadRV0Di1Qt",
  "artist_name": "Rick Astley",
  "primary_genre": "pop",
  "artist_popularity": 73,
  "follower_count": 7842910
}
```

**Response `404`** — artist not found

```json
{
  "detail": "Artist '0gxyHStUsqpMadRV0Di1Qt' not found"
}
```

**Response `503`** — gold data not loaded

```json
{
  "detail": "Artist data not loaded"
}
```

**Example**

```bash
curl http://localhost:8001/artists/0gxyHStUsqpMadRV0Di1Qt
```

---

## 6. Error Responses

| Status | When it occurs |
|---|---|
| `404 Not Found` | The requested `track_id` or `artist_id` does not exist in the gold layer |
| `422 Unprocessable Entity` | A query parameter fails validation (e.g. `n=0` or a malformed `liked_track_ids` string) |
| `503 Service Unavailable` | The gold-layer Parquet files or the trained model have not been loaded — see the `detail` field for the exact fix |

All error bodies follow the FastAPI default shape:

```json
{
  "detail": "Human-readable description of what went wrong"
}
```

---

## 7. Configuration

The API reads its settings from `config/config.yaml` at startup. The relevant keys are:

| Config key | Default | Description |
|---|---|---|
| `storage.gold_dir` | `data/gold` | Directory containing the gold Parquet files |
| `models.models_dir` | `models` | Directory containing the trained model artifacts |
| `models.collab_weight` | `0.7` | Blend weight for collaborative filtering (0.0–1.0); the remainder goes to content-based |

To override defaults, edit `config/config.yaml`. Values outside valid ranges are clamped and a warning is logged at startup.
