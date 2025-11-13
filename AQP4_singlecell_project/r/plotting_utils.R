ggBackground <- function(p, object, reduction = "umap", size = c(5, 4)) {
  data <- DimPlot(object, reduction = reduction)$data

  p <- p +
    scattermore::geom_scattermore(
      data = data,
      mapping = aes(
        x = !!sym(names(data)[1]),
        y = !!sym(names(data)[2])
      ),
      color = "black",
      pointsize = size[1]
    ) +
    scattermore::geom_scattermore(
      data = data,
      mapping = aes(
        x = !!sym(names(data)[1]),
        y = !!sym(names(data)[2])
      ),
      color = "white",
      pointsize = size[2]
    )
  return(p)
}

plot_celltype_proportions <- function(integrated_new) {
  plot_df <- integrated_new@meta.data %>%
    as.data.frame() %>%
    dplyr::group_by(type, celtype_level1) %>%
    dplyr::summarize(n = dplyr::n(), .groups = "drop")

  ggplot(plot_df) +
    geom_col(
      mapping = aes(x = celtype_level1, y = n, fill = type)
    ) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(colour = "black", size = 0.5),
      axis.text.x = element_text(
        colour = "black",
        angle = 90, vjust = 0.5, hjust = 1
      ),
      axis.text.y = element_text(colour = "black")
    )
}
