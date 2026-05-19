{{ config(materialized='view') }}

select *
from read_parquet('{{ var("silver_dir") }}/audio_features.parquet')
