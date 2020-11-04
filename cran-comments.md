## ML2Pvae Package Information
This is the first submission of this package to CRAN. A few notes about examples and tests: we use \donttest{} for examples because all functions use Tensorflow (which requires Python installations). This is not possible to configure for CRAN automatic checks, so we don't include unit tests, don't run examples, and use a static vignette.

## Test environments
local Windows 10 ; R 3.6.0
https://win-builder.r-project.org/upload.aspx ; R-release, R-devel, R-oldrelease

## R CMD check results
0 errors, 0 warnings, 3 notes
  NOTE: New submission
  NOTE: Non-standard files/directories found at top level:
    'cran-comments.md'
  -Explanation: cran-comments.md is for the CRAN reviewers, will later be added to .Rbuildignore
  NOTE: .onLoad calls: packageStartupMessage("..")
    See section 'Good practice' in '?.onAttach'
  -Explanation: My use of packageStartupMessage("..") seems to coincide with the 'Good practice' suggestions - most resources online say to ignore this note

## Resubmission
This is the second submission of this package. In this version I have:
 - Changed the title to have less than 65 characters
 - Wrote references in the description of DESCRIPTION file in the form: "authors (year) <doi:...>"
 - changed the names of two functions: build_vae_standard_normal() --> build_vae_independent() and build_vae_normal_full_covariance() --> build_vae_correlated()