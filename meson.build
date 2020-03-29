project('mir-blas', 'd', 'c', version: '1.1.10', license: 'BSL-1.0')

description = 'Mir BLAS - ndslice wrapper for BLAS'

cc = meson.get_compiler('c')
blas_deps = [cc.find_library('openblas', required : false)]
if not blas_deps[0].found()
    blas_deps = [cc.find_library('blas')]
    cblas_dep = cc.find_library('cblas', required : false)
    if cblas_dep.found()
        blas_deps += cblas_dep
    endif
endif

subprojects = ['mir-core', 'mir-algorithm']

has_cpp_headers = false

sources_list = [
    'mir/blas',
]

sources = []
foreach s : sources_list
    sources += 'source/' + s + '.d'
endforeach

add_project_arguments([
    '-preview=dip1008',
    '-lowmem',
], language: 'd')

required_deps = blas_deps + [
    dependency('cblas-d', fallback : ['cblas', 'cblas_dep']),
]

foreach p : subprojects 
    required_deps += dependency(p, fallback : [p, p.underscorify() + '_dep'])
endforeach

foreach p : subprojects 
    required_deps += dependency(p, fallback : [p, p.underscorify() + '_dep'])
endforeach

directories = ['source']

if has_cpp_headers
    directories += 'include'
endif

directories = include_directories(directories)

this_lib = library(meson.project_name(),
    sources,
    include_directories: directories,
    install: true,
    version: meson.project_version(),
    dependencies: required_deps,
)

this_dep = declare_dependency(
    link_with: [this_lib],
    include_directories: directories,
    dependencies: required_deps,
)

test_versions = []

if has_cpp_headers
    install_subdir('include/',
        strip_directory :true,
        install_dir: 'include/',
    )
endif

install_subdir('source/',
    strip_directory : true,
    install_dir: 'include/d/' + meson.project_name(),
)

import('pkgconfig').generate(this_lib,
    description: description,
    subdirs: 'd/' + meson.project_name(),
)

mir_blas_dep = this_dep
mir_blas_lib = this_lib

test_subdirs = ['example']
