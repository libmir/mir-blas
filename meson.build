project('mir-blas', 'd', 'c', version : '1.0.0', license: 'BSL-1.0')

mir_algorithm_dep = dependency('mir-algorithm', fallback : ['mir-algorithm', 'mir_algorithm_dep'])
cblas_d_dep = dependency('cblas-d', fallback : ['cblas', 'cblas_dep'])
cc = meson.get_compiler('c')
blas_deps = [cc.find_library('openblas', required : false)]
if not blas_deps[0].found()
    blas_deps = [cc.find_library('blas')]
    cblas_dep = cc.find_library('cblas', required : false)
    if cblas_dep.found()
        blas_deps = [blas_deps, cblas_dep]
    endif
endif

required_deps = [blas_deps, cblas_d_dep, mir_algorithm_dep]

mir_blas_dir = include_directories('source/')

mir_blas_src = [
    'source/mir/blas.d',
]

mir_blas_lib = library(meson.project_name(),
    mir_blas_src,
    include_directories: mir_blas_dir,
    install: true,
    version: meson.project_version(),
    dependencies: required_deps,
)

mir_blas_dep = declare_dependency(
    link_with: [mir_blas_lib],
    include_directories: mir_blas_dir,
    dependencies: required_deps,
)

install_subdir('source/',
    strip_directory : true,
    install_dir: 'include/d/' + meson.project_name(),
)

import('pkgconfig').generate(
    name: meson.project_name(),
    description: 'ndslice API for BLAS.',
    subdirs: 'd/' + meson.project_name(),
    libraries: [mir_blas_lib],
    version: meson.project_version(),
)

if get_option('with_test')

    mir_blas_test_exe = executable(meson.project_name() + '-test',
        mir_blas_src,
        include_directories: mir_blas_dir,
        d_unittest: true,
        d_module_versions: ['mir_test'],
        link_args: '-main',
        dependencies: required_deps,
    )

    test(meson.project_name() + '-test', mir_blas_test_exe)

endif
