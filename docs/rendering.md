# Rendering

To facilitate inspecting segmentation results, Ais comes with a built-in isosurface renderer that is similar to the surface renderer in ChimeraX. As soon as an Export job has created a segmentation, these results are available in the 'Render' tab of the main menu.

By default, Ais looks for segmentation results in the same folder in which the original `.mrc` file is located. Specify a different directory by clicking the '...' button or editing the path in the 'Volumes' tab of the main menu.

<figure markdown="span">
  ![Rendering segmentation results of 10 different models in the built-in renderer](res/render_1.PNG){ .with-border }
  <figcaption>Figure 10 – Rendering segmentation results of 10 different models in the built-in renderer.</figcaption>
</figure>

3D models can be ported from Ais directly into ChimeraX, Blender, or saved using the `.obj` file format, using the buttons in the 'Export 3D scene' submenu.

In scNodes+Ais - the version of Ais integrated into our main superCLEM software suite, available at [github.com/bionanopatterning/scNodes](https://github.com/bionanopatterning/scNodes) - additional functionality is available to render fluorescence overlays in the segmentation editor. Tomograms with fluorescence overlays can be prepared in the scNodes' Correlation Editor and can be directly forwarded into Ais with a single button click. These overlays can then be consulted during every step of the segmentation workflow.

<figure markdown="span">
  ![An example of scNodes+Ais segmenting a correlated fluorescence and cryoET dataset](res/scnodes_pom_2d.png){ .with-border }
  <figcaption>Figure 11 – An example of scNodes+Ais, where we segment a correlated dataset of rsEGFP2-Vimentin single molecule localization fluorescence and cryoET. The fluorescence data guides the identification of vimentin filaments.</figcaption>
</figure>

<figure markdown="span">
  ![A 3D render of segmented ribosomes, membranes, and vimentin filaments with a fluorescence overlay](res/scnodes_pom_3d.png){ .with-border }
  <figcaption>Figure 12 – A 3D render of the same data as before, after segmenting ribosomes, membranes, and vimentin filaments. The (2D) fluorescence overlay is rendered using ray tracing and projection into the 3D volume of the tomogram.</figcaption>
</figure>
