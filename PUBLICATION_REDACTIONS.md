# Publication credential redactions

Detected credential tokens, credential assignments, and private-key material were replaced with placeholders before publication.
Some matches may be example credentials from collected third-party source code. No credentials were tested against external services.
Original copies of changed files are stored locally in the ignored `.local-publication-backup/` directory. That directory is not part of this repository.
The publication commit has no parents, so the previous local Git history containing hard-coded credentials is not uploaded.

| File | Replacements |
| --- | ---: |
| `code/A - step 2 - repo_summary.ipynb` | 1 |
| `code/get_repo_details_from_files.ipynb` | 8 |
| `code/topic_modeling_results/cleaned_df_with_topics_primary_use_cases_keyword.csv` | 1 |
| `code/topic_modeling_results/cleaned_df_with_topics_primary_use_cases_sentence.csv` | 1 |
| `code/topic_modeling_results/cleaned_df_with_topics_repository_intent_keyword.csv` | 1 |
| `code/topic_modeling_results/cleaned_df_with_topics_repository_intent_sentence.csv` | 1 |
| `code/topic_modeling_results/cleaned_df_with_topics_system_design_keyword.csv` | 1 |
| `code/topic_modeling_results/cleaned_df_with_topics_system_design_sentence.csv` | 1 |
| `code/topic_modeling_results/cleaned_df_with_topics_technical_focus_keyword.csv` | 1 |
| `code/topic_modeling_results/cleaned_df_with_topics_technical_focus_sentence.csv` | 1 |
| `dataset/Extracted_models/lmdeploy_extracted_models.csv` | 15 |
| `dataset/archived_old_code_content_processed_results/lmdeploy_all_info.csv` | 15 |
| `dataset/code_content/code_content_concated_files_lmdeploy.csv` | 11 |
| `dataset/code_content/code_content_concated_files_openllm.csv` | 1 |
| `dataset/code_content/code_content_concated_files_text_generation.csv` | 13 |
| `dataset/code_content_processed_results/lmdeploy_all_info.csv` | 15 |
| `dataset/code_content_processed_results/vllm_all_info_cleaned.csv` | 62 |
| `dataset/code_content_processed_results/vllm_all_info_cleaned_with_imports.csv` | 62 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_1.csv` | 7 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_10.csv` | 7 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_11.csv` | 8 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_12.csv` | 8 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_13.csv` | 9 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_14.csv` | 13 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_15.csv` | 7 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_16.csv` | 6 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_17.csv` | 3 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_2.csv` | 12 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_3.csv` | 11 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_4.csv` | 8 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_5.csv` | 11 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_6.csv` | 12 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_7.csv` | 7 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_8.csv` | 7 |
| `dataset/repo_info/old repo info/langchain/info_langchain_chunk_9.csv` | 13 |
| `dataset/repo_info/old repo info/openllm/info_openllm_chunk_1.csv` | 2 |
| `dataset/repo_info/old repo info/ray serve/info_ray_chunk_1.csv` | 2 |
| `dataset/repo_info/old repo info/ray serve/info_ray_chunk_2.csv` | 3 |
| `dataset/repo_info/old repo info/ray serve/info_ray_chunk_3.csv` | 2 |
| `dataset/repo_info/old repo info/sglang/info_sglang_chunk_1.csv` | 1 |
| `dataset/repo_info/old repo info/sglang/info_sglang_chunk_2.csv` | 2 |
| `dataset/repo_info/old repo info/sglang/info_sglang_chunk_4.csv` | 1 |
| `dataset/repo_info/old repo info/tensorrt_llm/info_tensorrt_llm_chunk_1.csv` | 50 |
| `dataset/repo_info/old repo info/tensorrt_llm/info_tensorrt_llm_chunk_2.csv` | 11 |
| `dataset/repo_info/old repo info/vllm/cleaned_info_vllm_chunk_2.csv` | 1 |
| `dataset/repo_info/old repo info/vllm/info_vllm_chunk_1.csv` | 12 |
| `dataset/repo_info/old repo info/vllm/info_vllm_chunk_2.csv` | 1 |
| `dataset/repo_info/old repo info/vllm/vllm_repo_summary_all_chunks.csv` | 1 |
| `dataset/repo_info/old repo info/vllm/vllm_repo_summary_chunk_2.csv` | 1 |
| `dataset/repo_info/old repo info/vllm/vllm_repo_summary_keywords_all_chunks.csv` | 1 |
| `dataset/repo_info/old repo info/vllm/vllm_repo_summary_keywords_chunk_2.csv` | 1 |

Total: 51 files, 452 replacements.

## Large dataset publication

Five additional large datasets were scanned before publication through Git LFS. Original files are retained in the ignored `.local-publication-backup/large-files-20261108/` directory.

| File | Replacements |
| --- | ---: |
| `dataset/Extracted_models/vllm_extracted_models.csv` | 341 |
| `dataset/code_content/code_content_concated_files_sglang.csv` | 13 |
| `dataset/code_content/code_content_concated_files_vllm.csv` | 375 |
| `dataset/code_content_processed_results/vllm_all_info.csv` | 242 |
| `dataset/code_content_with_imports_asnames_subimports/sglang.csv` | 13 |
