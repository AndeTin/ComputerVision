"""HW01 stage scripts, one per assignment question.

    q1_input_pipeline        reproducible input pipeline, split rule, EXIF, stats
    q2_resolution_comparison geometry strategy and interpolation comparison
    q3_augmentation_design   M0 baseline, targeted augmentation, M1 ratio sweep

Each module is runnable on its own and exposes ``run()`` plus a
``_print_summary()`` used by the entry point. Importing a stage does not run
any experiment; the expensive work happens inside ``run()``.
"""
