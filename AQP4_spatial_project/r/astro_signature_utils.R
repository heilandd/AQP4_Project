load_mouse_astro_signature <- function(path) {
  AstroSig <- read.csv(path, sep = ";")
  convert  <- human2mouse(AstroSig$gene)

  AstroSig %>%
    dplyr::left_join(
      data.frame(gene = convert$human, mouse = convert$mice),
      by = c("gene" = "gene")
    ) %>%
    dplyr::mutate(gene = mouse) %>%
    dplyr::select(gene, ont)
}
