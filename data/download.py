from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="HuggingFaceH4/aime_2024",
    filename="data/train-00000-of-00001.parquet",
    repo_type="dataset",
    local_dir="."
)