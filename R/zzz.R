#' Display a message upon loading package
#' 
#' @param libnam the library name
#' @param pkgname the package name
.onLoad <- function(libnam, pkgname){
  packageStartupMessage("Thank you for installing ML2Pvae.\nBe sure that the following Python libraries have been installed:
    tensorflow
    keras
    tensorflow-probability\nWe also recommend using the developer, rather than CRAN, version of keras for R.\nThis can be installed with devtools::install_github('rstudio/keras')")
  reticulate::configure_environment(pkgname)
}