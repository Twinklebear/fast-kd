# fast-kd

A pretty fast k-d tree builder which combines a coarse bottom-up build
with parallel top-down construction of treelets within the leaves of
the coarse tree.

## Dependencies

fast-kd requires TBB, SDL2 and GLM, which you can specify the paths to
through CMake:

```
cmake <args> -DTBB_DIR=<path> -DSDL2_DIR=<path to SDL2 root> \
    -Dglm_DIR=<path to glmConfig.cmake>
```

On Linux these may be found automatically by CMake.

