plot_spatial_overview <- function(spata_obj) {
  plotSurface(spata_obj, pt_alpha = 0) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(colour = "black", size = 0.5),
      axis.text.x      = element_text(colour = "black"),
      axis.text.y      = element_text(colour = "black")
    ) +
    coord_fixed() +
    theme(
      legend.text     = element_text(size = 6),
      legend.key.size = unit(0.3, "cm"),
      legend.title    = element_text(size = 8)
    ) +
    SPATA2::ggpLayerAxesSI(spata_obj) +
    xlab("Dimension x-space [cm]") +
    ylab("Dimension y-space [cm]")
}

plot_gene_spatial <- function(spata_obj, gene, limits = c(0, 0.4)) {
  plotSurface(spata_obj, color_by = gene, alpha_by = gene) +
    scale_color_gradientn(
      colors = viridis::inferno(50),
      limits = limits,
      oob    = scales::squish
    ) +
    scale_alpha(limits = limits, oob = scales::squish) +
    guides(color = guide_colourbar(
      barwidth = 0.3, barheight = 8, ticks = FALSE, frame.colour = "black"
    ), label = FALSE) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(colour = "black", size = 0.5),
      axis.text.x      = element_text(colour = "black"),
      axis.text.y      = element_text(colour = "black")
    ) +
    coord_fixed() +
    theme(
      legend.text     = element_text(size = 6),
      legend.key.size = unit(0.3, "cm"),
      legend.title    = element_text(size = 8)
    ) +
    SPATA2::ggpLayerAxesSI(spata_obj) +
    xlab("Dimension x-space [cm]") +
    ylab("Dimension y-space [cm]")
}

plot_single_cell_voronoi <- function(spata_obj, cell_types, color_map) {
  p <- plotSurface(spata_obj, pt_alpha = 0)

  p +
    ggforce::geom_voronoi_tile(
      data    = cell_types,
      mapping = aes(x = x_sc, y = y_sc, group = -1L, fill = CellType),
      max.radius = 2,
      colour = "black",
      linewidth = 0.1
    ) +
    scale_fill_manual(values = color_map) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(colour = "black", size = 0.5),
      axis.text.x      = element_text(colour = "black"),
      axis.text.y      = element_text(colour = "black")
    ) +
    coord_fixed() +
    theme(
      legend.text     = element_text(size = 6),
      legend.key.size = unit(0.3, "cm"),
      legend.title    = element_text(size = 8)
    ) +
    SPATA2::ggpLayerAxesSI(spata_obj) +
    xlab("Dimension x-space [cm]") +
    ylab("Dimension y-space [cm]")
}

plot_prediction_scatter <- function(res) {
  library(scales)
  ggplot(res) +
    geom_point(aes(
      x = jitter(scales::rescale(logit.1, c(0, 1)), 100),
      y = scales::rescale(region, c(0, 1)),
      color = scales::rescale(region, c(0, 1))
    )) +
    scale_color_gradientn(colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(colour = "black", size = 0.5),
      axis.text.x      = element_text(colour = "black"),
      axis.text.y      = element_text(colour = "black")
    ) +
    coord_fixed() +
    theme(
      legend.text     = element_text(size = 6),
      legend.key.size = unit(0.3, "cm"),
      legend.title    = element_text(size = 8)
    ) +
    guides(color = guide_colourbar(
      barwidth = 0.3, barheight = 8, ticks = FALSE, frame.colour = "black"
    ), label = FALSE)
}

plot_prediction_rank <- function(res) {
  library(scales)
  ggplot(res %>% arrange(logit.1)) +
    geom_point(aes(
      x = 1:nrow(res),
      y = logit.1,
      color = scales::rescale(logit.1, c(0, 1))
    )) +
    scale_color_gradientn(colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(colour = "black", size = 0.5),
      axis.text.x      = element_text(colour = "black"),
      axis.text.y      = element_text(colour = "black")
    ) +
    theme(
      legend.text     = element_text(size = 6),
      legend.key.size = unit(0.3, "cm"),
      legend.title    = element_text(size = 8)
    ) +
    guides(color = guide_colourbar(
      barwidth = 0.3, barheight = 8, ticks = FALSE, frame.colour = "black"
    ), label = FALSE)
}
