file(REMOVE_RECURSE
  "libOrchestra.a"
  "libOrchestra.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/Orchestra.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
