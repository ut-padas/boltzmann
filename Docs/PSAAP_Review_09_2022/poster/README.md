#### Pecos Poster Template


This simple LaTeX example assumes that relevant logo images are housed in a
`shared_figures/` subdirectory. If working on a new poster directly in the
pecos GitHub repository, the recommendation is to make a soft link to the
`pecos/slides/shared_figures` directory as opposed to making separate copies of
the images.

Similarly, to use the baposter format, link `baposter.cls` file from
`pecos/slides/latex_poster_template` into your local directory, and
start your poster from `pecos/slides/latex_poster_template/poster.tex`
(for consistent colors, title formatting, etc). An example of this is setup is used
[here](https://github.com/pecos/pecos/tree/master/slides/2022/csem_open_house).

The example is also setup to search for images in a `figures/` subdirectory, so
this would be a good place to store images specific to your poster.

