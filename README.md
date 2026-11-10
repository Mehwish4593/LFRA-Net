# LFRA-Net

A lightweight retinal vessel segmentation network, **174,097 parameters**,
trained and evaluated on DRIVE, STARE and CHASE_DB1.

> M. Mehmood et al., *LFRA-Net: A Lightweight Network with Focal Modulation and
> Region-Aware Attention for Retinal Vessel Segmentation*, DICTA 2025.

## Contents

    lfranet.py             the model, as a standalone module
    predict.py             run a checkpoint over a folder of images
    evaluate.py            metrics in Python, matching Prediction_code/
    Prediction_code/       the original MATLAB evaluation (reference)
    Weights/               trained weights, see below
    results/               the JSON behind the table in this README
    LFRA-Net.ipynb         the training notebook
    requirements.txt

## Results

Held-out test images, scored with the protocol in `Prediction_code/Evaluation.m`:
a per-image threshold maximising F1 against that image's own mask, then
`bwareaopen(out, 50)`, over the full frame with no field-of-view mask.

Figures use 8-way test-time augmentation: the image is predicted at four
rotations and two flips and the maps are averaged.

| dataset | Dice | AP |
|---|---|---|
| DRIVE | 0.8229 | 0.9107 |
| STARE | 0.8525 | 0.9380 |
| CHASE | 0.8542 | 0.9323 |

Average precision is the area under the precision-recall curve. It needs no
threshold, so it measures the ranking rather than the operating point.

    python evaluate.py --weights Weights/Vseg-weights_Combined_DRIVE.h5 \
                       --images path/to/test/images --gt path/to/test/GT --tta

## Weights

| file | trained on | DRIVE | STARE | CHASE |
|---|---|---|---|---|
| `Vseg-weights_Combined_DRIVE.h5` | DRIVE | **0.8199** | | |
| `Vseg-weights_Combined_STARE.h5` | STARE | | **0.8502** | |
| `Vseg-weights_Combined_CHASE.h5` | CHASE_DB1 | | | **0.8467** |
| `Vseg-weights_Combined.h5` | all three | 0.8184 | 0.8465 | 0.8440 |

Single-pass Dice. The combined model is within 0.004 of each dataset-specific
one on its own test set, so a single set of weights covers all three at
negligible cost.

## Architecture

An encoder-decoder with a focal-modulation bottleneck and region-aware attention
on the skip pathways.

**conv_block.** Three parallel convolutions over the same input, 1x1, 3x3, and
3x3 dilated by 2, all without bias, summed, then batch norm, dropout and
LeakyReLU(0.1); then a second 3x3 convolution with the same batch norm, dropout
and LeakyReLU. Widths are 8, 16 and 32.

**down_block / up_block.** Stride-2 2x2 convolution and transposed convolution
respectively, each followed by batch norm and LeakyReLU(0.1).

**Focal modulation context aggregation** (the bottleneck). Three parallel
convolutions of kernel 1, 3 and 5 over the input, each multiplied by its own
1x1 sigmoid gate computed from that same input, then summed. The sum passes
through the focal modulation block, and the result is multiplied by a 1x1
projection of the input (query times modulator) and added back as a residual.

**Focal modulation block.** The difference between the input's global maximum
and global mean, scaled by `alpha = 0.25`, through a 1x1 sigmoid, multiplied
into the input and raised elementwise to `gamma = 2`. Both statistics are
global, so the modulation is per channel.

**Region-aware attention** (`ana_attention_block`). A 3x3 convolution, batch
norm and ReLU, from which a channel descriptor is formed by multiplying the
global max and global average pools. The descriptor rescales the spatially
averaged feature map, and the result gates the block's input.

**Skip pathways.** Rather than passing encoder features straight across, the
decoder receives `concat(ana_attention(c2), up(c3))` at one scale and
`concat(ana_attention(c1), up(c2))` at the next, so each skip already carries
information from the stage below it.

## Install

    pip install -r requirements.txt

The weights are Keras 2 `.h5` files, so the model is built under `tf_keras`
(legacy Keras) rather than Keras 3.

## Predict

    python predict.py --weights Weights/Vseg-weights_Combined_DRIVE.h5 \
                      --images path/to/test/images --out predictions/

Writes one 8-bit PNG per input as `<name>_Pred.png`, which is the layout
`Prediction_code/Evaluation.m` expects.

Preprocessing is the notebook's: `skimage.transform.resize` to 512x512 with
`preserve_range`, then divide by 255. skimage anti-aliases when downsampling and
PIL's `BILINEAR` does not, so substituting one for the other changes the result
on DRIVE's 584x565 inputs.

## Evaluate

The reference implementation is MATLAB, in `Prediction_code/`. Run `predict.py`
first, put `<n>_Pred.png` and `<n>_GT.png` in one folder, set `path` at the top
of `Evaluation.m` and run it. `evaluate.py` is a Python port of the same
protocol.

One thing worth knowing when quoting numbers: the per-image threshold is chosen
against that image's own ground truth. It is the convention in much of the
retinal vessel literature and it is what the paper reports, but it is an upper
bound rather than a deployment figure, since a running system has no mask to
tune against.

## Train

`LFRA-Net.ipynb` holds the training run. Data is loaded by `get_data()`, which
expects a folder containing `images/` and `GT_BV/`.

## Citation

    @inproceedings{mehmood2025lfranet,
      title     = {LFRA-Net: A Lightweight Network with Focal Modulation and
                   Region-Aware Attention for Retinal Vessel Segmentation},
      author    = {Mehmood, Mehwish and others},
      booktitle = {Digital Image Computing: Techniques and Applications (DICTA)},
      year      = {2025}
    }
