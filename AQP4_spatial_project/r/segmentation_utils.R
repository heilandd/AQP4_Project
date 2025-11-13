add_spatial_segmentation <- function(spata_obj) {
  spata_obj <- SPATA2::createSpatialSegmentation(spata_obj)
  return(spata_obj)
}
