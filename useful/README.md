# Optimize

1. Run `align_prod.py optimize --round=0`. This will create initial `*.scale.txt` files.
2. Run `align_prod.py combine --round=0`. This will create the initial `global_scale.txt`
3. Run `align_prod.py optimize --round=1`. This will do the actual calling and append to `*.scale.txt` files. The file now has two lines.
4. Run `align_prod.py combine --round=1`. This will calculate the deviations of each channel from the first line of global_scale and the second line of `*.scale.txt` and append to `global_scale.txt`. The file now has two lines.

The round number should be the number of lines in the scale file **before** you run it.
