{{ config(materialized='view') }}

select
    track_id,
    name,
    cast(popularity as integer)        as track_popularity,
    duration_ms,
    explicit,
    primary_artist_id,
    primary_artist_name,
    album_id,
    album_name,
    release_date
from read_parquet('{{ var("silver_dir") }}/tracks.parquet')
