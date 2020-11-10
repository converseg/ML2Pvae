## ML2Pvae Package Information
This is the first submission of this package to CRAN. A few notes about examples and tests: we use \donttest{} for examples because all functions use Tensorflow (which requires Python installations). This is not possible to configure for CRAN automatic checks, so we don't include unit tests, don't run examples, and use a static vignette.

## Test environments
local Windows 10 ; R 3.6.0
https://win-builder.r-project.org/upload.aspx ; R-release, R-devel, R-oldrelease

## R CMD check results
0 errors, 0 warnings, 1 note
  NOTE: New submission

## Resubmission 2020-10-28
This is the second submission of ML2Pvae to CRAN. Comments from reviewer Julia Haider:
```
Please reduce the length of the title to less than 65 characters.

Please write references in the description of the DESCRIPTION file in
the form
authors (year) <doi:...>
authors (year) <arXiv:...>
authors (year, ISBN:...)
or if those are not available: authors (year) <https:...>
with no space after 'doi:', 'arXiv:', 'https:' and angle brackets for
auto-linking.
(If you want to add a title as well please put it in quotes: "Title")

Please fix and resubmit.

Best,
Julia Haider
```
In this submission I have:
 - Changed the title to have less than 65 characters
 - Wrote references in the description of DESCRIPTION file in the form: "authors (year) <doi:...>"
 - Changed the names of two functions: build_vae_standard_normal() --> build_vae_independent() and build_vae_normal_full_covariance() --> build_vae_correlated()
 
## Resubmission 2020-11-10
Comments from reviewer Gregor Seyer:
```
Please always write package names, software names and API (application
programming interface) names in single quotes in title and description.
e.g: --> 'tensorflow'
Please note that package names are case sensitive.

All your examples are wrapped in \donttest{} and therefore do not get
tested.
Please unwrap the examples if that is feasible and if they can be
executed in < 5 sec for each Rd file or create additionally small toy
examples to allow automatic testing.

Please fix and resubmit.

Best,
Gregor Seyer
```
In this submission, I have made the following changes:
 - Put package/software/API names in single quotes in the title and description
 - It is not feasible to unwrap the examples from \donttest{}. The examples are all wrapped in \donttest{} because all functions use TensorFlow (which requires Python installations). This is not possible to configure for CRAN automatic checks, so we don't include unit tests, don't run examples, and use a static vignette.
 - I have also added a few other authors as contributors
 
