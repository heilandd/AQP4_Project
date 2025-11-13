get_project_root <- function() {
  root <- Sys.getenv("AQP4_PROJECT_ROOT", unset = NA)
  if (is.na(root)) {
    stop("Please set AQP4_PROJECT_ROOT environment variable to your project root.")
  }
  return(root)
}
