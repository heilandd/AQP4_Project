# AQP4 Single-cell/Spatial Transcriptomic Analysis

This repository contains a reproducible version of the AQP4 single-cell RNA-seq
and spatial transcriptomics analysis pipeline.

## Structure

```text


AQP4_spatial_project/
├─ data/
│  ├─ file_system.RDS
│  ├─ astro_signature.csv
│  ├─ colors_celltypes.RDS
│  ├─ VAE_ep200_fullData.h5ad
│  └─ VAE_ep250_fullData/
├─ r/
│  ├─ setup_spatial_project.R
│  ├─ plotting_utils.R
│  ├─ astro_signature_utils.R
│  ├─ segmentation_utils.R
│  ├─ ecosystem_utils.R
└─ python/
   ├─ scvi_preprocessing.py
   ├─ gnn_models.py
   ├─ gnn_training.py
   ├─ graph_construction.py
   └─ evaluation_utils.py


AQP4_singlecell_project/
├─ AQP4_singlecell_analysis.Rmd   # Main R Markdown analysis
├─ r/
│  ├─ setup_spatial_project.R     # Project root helper (env-based)
│  ├─ plotting_utils.R            # ggBackground + plotting helpers
│  └─ singlecell_utils.R          # Single-cell specific helpers (Azimuth, etc.)
├─ python/
│  └─ singlecell_attention.py     # Attention-based perturbation model
├─ data/
│  ├─ integrated_data_new.h5ad        # integrated AnnData (not included)
│  ├─ azimuth_core_GBmap.rds          # Azimuth reference (not included)
│  ├─ colors_cell_deconv.RDS          # reference colors (optional)
│  ├─ GBM_Neuron_colors.RDS           # reference colors for GBM neurons
│  └─ singlecell_raw/
│      ├─ AQP4KO/                     # 10x folders for KO samples
│      └─ AQP4WT/                     # 10x folders for WT samples
└─ results/                           # Not included

```

## Setup

1. Clone the repository and set the project root as an environment variable in R:

   ```r
   Sys.setenv(AQP4_PROJECT_ROOT = "/path/to/AQP4_singlecell_project")
   ```

2. Create and activate a conda environment for single-cell analysis (optional but recommended):

   ```bash
   conda create -n Single_cell r-base python scanpy scvi-tools pytorch -c conda-forge -c pytorch
   conda activate Single_cell
   ```

3. Install required R packages in that environment:

   ```r
   install.packages(c(
     "Seurat", "SPATA2", "harmony", "tidyverse", "readxl",
     "scattermore", "Azimuth", "RColorBrewer", "DOSE",
     "enrichplot", "clusterProfiler"
   ))
   ```

4. Install Python dependencies (inside the same conda env):

   ```bash
   pip install scanpy scvi-tools torch torchvision torchaudio tqdm
   ```

## Data expectations

- `data/integrated_data_new.h5ad`  
  Integrated AnnData object with:
  - `adata.obsm["X_umap"]` – 2D UMAP coordinates
  - `adata.obsm["X_scVI"]` – scVI latent space
  - `adata.obs["batch"]` – batch labels

- Raw 10x data for individual KO and WT samples organized as:

  ```text
  data/singlecell_raw/AQP4KO/<sample_id>/filtered_feature_bc_matrix.h5
  data/singlecell_raw/AQP4WT/<sample_id>/filtered_feature_bc_matrix.h5
  ```

- Azimuth reference object `data/azimuth_core_GBmap.rds` and associated color
  references `data/colors_cell_deconv.RDS`, `data/GBM_Neuron_colors.RDS`.
  These can be replaced by your own references; adjust paths in
  `AQP4_singlecell_analysis.Rmd` or `r/singlecell_utils.R` accordingly.

## Running the analysis

1. Open `AQP4_singlecell_analysis.Rmd` in RStudio or run it via `rmarkdown`:

   ```r
   rmarkdown::render("AQP4_singlecell_analysis.Rmd")
   ```

2. The notebook will:

   - Load integrated single-cell data (`integrated_data_new.h5ad`).
   - Build Seurat objects from raw 10x KO / WT batches.
   - Map mouse genes to human symbols for Azimuth projection.
   - Run Azimuth using a GBM reference to obtain cell-type annotations.
   - Perform Harmony integration, clustering, and UMAP embedding.
   - Attach scVI latent / UMAP from the Python pipeline.
   - Train an attention-based neural network to model WT vs KO perturbations.
   - Generate UMAP visualizations and cell-type composition plots.

3. Results include:

   - A processed Seurat object `results/integrated_new.RDS`.
   - Optional DE / GSEA tables written into `results/tables/` if you extend
     the notebook with the DE sections from your original analysis.

## Custom functions

All custom functions originally defined inside the analysis script have been
moved into:

- `r/plotting_utils.R` – background UMAP plotting (`ggBackground`) and
  cell-type proportion plots.
- `r/singlecell_utils.R` – helper functions for:
  - Reading batched 10x data (`read_10x_batches`),
  - Mouse→human gene mapping (`humanize_mouse_seurat`),
  - Azimuth integration (`run_azimuth_gbmap`),
  - Manual cell-type annotation (`build_manual_celltype_annotation`).
- `python/singlecell_attention.py` – attention network architecture and
  training routine.

This makes the R Markdown file more readable and easier to maintain.

## License

Add your preferred license here (e.g., MIT, GPL-3, or as required by your journal
or institution).
