
run("Memory & Threads...", "parallel=8");
run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=/fast2/thicc/thicc/down2/0 layout_file=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.5 max/avg_displacement_threshold=0.5 absolute_displacement_threshold=1.5 compute_overlap subpixel_accuracy computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=/fast2/thicc/thicc/down2");
