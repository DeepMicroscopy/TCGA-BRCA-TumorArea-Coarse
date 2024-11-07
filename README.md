# TCGA-BRCA-TumorArea-Coarse
This repository contains the following:
1. code used to run the experiments on the coarse annotations of tumor areas from the TCGA-BRCA dataset.
2. Code used to do inference on the held-out test set.
3. The '16x' downsampled coarsely annotated tumor area image files.
4. The '16x' downsampled finely annotated tumor area image files.
5. The .p file contains the '16x' scaled down polygon annotations organized in a dictionary with two main keys: 'incl_vec' and 'excl_vec'. Each key holds a dictionary where TIFF image filenames map to lists. The 'incl_vec' key represents polygon annotations for areas in the images that should be included as important, while 'excl_vec' contains annotations for areas to be excluded.
