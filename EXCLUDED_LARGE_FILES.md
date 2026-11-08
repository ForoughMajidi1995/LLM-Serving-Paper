# Files excluded from the GitHub upload

GitHub blocks regular Git files larger than 100 MiB (104,857,600 bytes).
Source: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

The files below remain on the local computer. No Git LFS files are uploaded in this step.

## Project files exceeding the limit

| File | Bytes | MiB |
| --- | ---: | ---: |
| `dataset/Extracted_models/vllm_extracted_models.csv` | 724,838,557 | 691.26 |
| `dataset/archived_old_code_content_processed_results/vllm_all_info.csv` | 703,430,598 | 670.84 |
| `dataset/archived_old_code_content_processed_results/vllm_all_info_cleaned.csv` | 166,547,398 | 158.83 |
| `dataset/code_content/code_content_concated_files_sglang.csv` | 288,732,463 | 275.36 |
| `dataset/code_content/code_content_concated_files_vllm.csv` | 1,839,715,720 | 1,754.49 |
| `dataset/code_content_processed_results/vllm_all_info.csv` | 350,404,075 | 334.17 |
| `dataset/code_content_with_imports_asnames_subimports/sglang.csv` | 310,650,914 | 296.26 |

Total: 7 project files, 4,384,319,725 bytes.

## Oversized files inside the excluded local Python environment

The entire `.venv/` directory is a local dependency installation and is excluded from the upload. These files also exceed the per-file limit:

| File | Bytes | MiB |
| --- | ---: | ---: |
| `.venv/lib/python3.11/site-packages/llvmlite/binding/libllvmlite.dylib` | 116,292,496 | 110.91 |
| `.venv/lib/python3.11/site-packages/torch/lib/libtorch_cpu.dylib` | 207,886,592 | 198.26 |

## Other local files excluded

Git metadata (`.git/`), the local Python environment (`.venv/`), macOS metadata (`.DS_Store`), Python bytecode caches (`__pycache__/` and `*.py[cod]`), and notebook checkpoints (`.ipynb_checkpoints/`) are excluded as local generated files. They are not project files excluded for size. The credential backup (`.local-publication-backup/`) is also kept locally and excluded from publication.
