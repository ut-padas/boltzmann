#### Pecos Talk Template


This simple LaTeX example assumes that relevant logo images are housed in a
`shared_figures/` subdirectory. If working on a new talk directly in the
pecos GitHub repository, the recommendation is to make a soft link to the
`pecos/slides/shared_figures` directory as opposed to making separate copies of
the images.

Similarly, to leverage a common Beamer template, consider linking the
`beamerthemeodenpecos.sty` file from `pecos/slides/latex_template` into your
local talk. An example of this is setup is used
[here](https://github.com/pecos/pecos/tree/master/slides/2021/torchTeam_axisym_Bfield_update).

The example is also setup to search for images in a `figures/` subdirectory, so
this would be a good place to store images specific to your talk.

A pdf build of this example is available [here](https://github.com/pecos/pecos/blob/master/slides/latex_template/talk.pdf).
