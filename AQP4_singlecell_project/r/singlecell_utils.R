importPyColors <- function() {
  # Placeholder: adapt to your preferred color palette.
  # Returns a list, as in your original environment.
  tab10 <- RColorBrewer::brewer.pal(8, "Set2")
  tab20 <- grDevices::rainbow(20)
  list(tab10, tab20)
}

read_10x_batches <- function(folder, type_label) {
  paths <- dir(folder, full.names = TRUE)
  batch_names <- dir(folder)
  map(seq_along(paths), function(i) {
    z <- paths[i]
    obj <- Seurat::Read10X_h5(file.path(z, "filtered_feature_bc_matrix.h5")) %>%
      Seurat::CreateSeuratObject()
    obj$batch <- paste0("Batch_", batch_names[i])
    obj$type  <- type_label
    obj
  })
}

humanize_mouse_seurat <- function(integrated) {
  mouse_human_genes <- read.csv(
    "http://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",
    sep = "\t"
  )

  human.only <- mouse_human_genes %>%
    dplyr::select(DB.Class.Key, Common.Organism.Name, Symbol) %>%
    dplyr::filter(Common.Organism.Name == "human") %>%
    dplyr::rename(human = Symbol)

  mice.only <- mouse_human_genes %>%
    dplyr::select(DB.Class.Key, Common.Organism.Name, Symbol) %>%
    dplyr::filter(Common.Organism.Name == "mouse, laboratory") %>%
    dplyr::rename(mice = Symbol)

  genes_mice <- data.frame(
    mice = rownames(integrated@assays$RNA@data)
  )

  genes_mice <- genes_mice %>%
    dplyr::left_join(mice.only, by = "mice") %>%
    dplyr::select(mice, DB.Class.Key) %>%
    dplyr::left_join(human.only, by = "DB.Class.Key") %>%
    dplyr::select(mice, human) %>%
    dplyr::filter(!is.na(human))

  mat <- Seurat::GetAssayData(integrated, "RNA", "counts")
  mat <- mat[genes_mice$mice, ]
  rownames(mat) <- genes_mice$human

  RNA <- Seurat::CreateAssayObject(mat)
  integrated_new <- Seurat::CreateSeuratObject(RNA)
  integrated_new@meta.data <- integrated@meta.data
  integrated_new
}

run_azimuth_gbmap <- function(
    integrated_new,
    az_ref_path,
    color_ref_path
) {
  library(Azimuth)
  source("r/modified_azimuth.R", local = TRUE)

  az_ref <- readRDS(az_ref_path)
  az_ref$annotation_level_4 <- az_ref$annotation_level_4 %>%
    str_replace_all(" ", "_") %>%
    str_replace_all("-", "_") %>%
    str_replace_all("/", "_")

  colors <- readRDS(color_ref_path)
  cc <- colors$colors
  names(cc) <- colors$annotation_level_4

  query <- Seurat::DietSeurat(integrated_new)
  DefaultAssay(query) <- "RNA"
  query@assays$SCT <- NULL

  ds.ref <- RunAzimuth(
    query,
    reference = az_ref,
    normalization.method = "LogNormalize",
    annotation.levels = c("annotation_level_3", "annotation_level_4")
  )

  list(query = integrated_new, reference = ds.ref)
}

build_manual_celltype_annotation <- function() {
  data.frame(
    seurat_clusters = factor(0:26),
    celtype_level1 = c(
      "MES_like",
      "NPC_like",
      "MG_pro_infl",
      "TAM_MHC",
      "OPC_like",
      "AC_like",
      "OPC_like",
      "Mono_anti_infl",
      "MG_aging_sig",
      "Astrocytes",
      "Monocytes_naive",
      "OPC_like",
      "T_cells",
      "cDCs",
      "Pericyte",
      "MG_aging_sig",
      "Endothel",
      "RG",
      "OPC_like",
      "Oligodendrocyte",
      "B_cell",
      "TAMs_hypoxia",
      "Monocytes_naive",
      "Mast",
      "T_Reg",
      "TAMs_hypoxia",
      "NPC_like"
    )
  )
}
