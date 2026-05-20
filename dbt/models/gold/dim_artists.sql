{% if target.type == 'duckdb' %}
{{ config(
    materialized='external',
    location=var('gold_dir') ~ '/dim_artists.parquet',
    format='parquet'
) }}
{% else %}
{{ config(materialized='table') }}
{% endif %}

select
    artist_id,
    artist_name,
    artist_popularity   as popularity,
    followers,
    genres,
    primary_genre
from {{ ref('stg_silver_artists') }}
