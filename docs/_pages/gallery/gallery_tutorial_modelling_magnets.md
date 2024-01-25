---
orphan: true
---

(gallery-tutorial-modelling-magnets)=

# Modelling a real magnet

Whenever you wish to compare Magpylib simulations with experimental data obtained using a real permanent magnet, you might wonder how to properly set up a Magpylib magnet object to reflect the physical permanent magnet in question.

The goal of this tutorial is to provide a better understanding of permanent magnets, how to extract the information you want from the respective datasheet, and how to set up the correct attributes for the `magpylib.magnet` instances.

## Kurzgesagt
With the datasheet of the permanent magnet in hands, look for the *working point* of the magnet: the point where the corresponding *permeance coefficient* line crosses the B-H curve. From there, trace a vertical line up and verify where the intrinsic curve gets intersected. That is the magnetic polarization value you should use in Magpylib simulations. Some datasheets may directly show this value, calling it the "remanence at  the working point" or "$B_r$ at the working point".

To know which of the several straight lines (corresponding to different permeance coefficients) you have to follow, check the number indicated where the line starts or ends. That number is related to the ratio between the length and width of the magnet.

If you wish to get a bit more of "background knowledge", just keep reading.
## Hysteresis Loop

If you've worked with magnetism, chances are very high that you have seen a hysteresis loop. This curve presents the behavior of the permanent magnet's *magnetic polarization* $J$ (or sometimes the *average magnetization* $M$, related through µ<sub>0</sub>) and the resulting *magnetic flux density* $B$ according to an external applied *magnetic field strength* $H$. While $B$ and $H$ exist everywhere, $J$ only exists inside the material. The relation between these quantities is $$B = µ_0H + J$$
The curve that describes the relationship between $J$ and $H$ is known as the *intrinsic curve*, whereas the relationship between $B$ and $H$ is known as the *normal curve*, also called B-H curve. The normal curve represents the resulting $B$, considering contributions both by the magnet's magnetic polarization $J$ and the external applied field $H$ (wording can get confusing in Magnetism, but I am confident you can follow).

::::{grid} 2
:::{grid-item}
:columns: 3
:::
:::{grid-item}
:columns: 6
![](../../_static/images/gallery_tutorial_magnet_hysteresis.png)
:::
::::

With the increase of $H$ starting in the first quadrant, the material's polarization will increase up to its maximum possible value, the saturation $J_S$. Higher values of $H$ will not affect $J$, while the resulting $B$ will keep increasing. This is reflected in the flat line of the intrinsic curve and the continuing higher normal curve. As the external field drops to zero, the material will retain its magnetization saturation while resulting $B$ decreases. The magnetic flux density value when $H = 0$ is known as the *remanence field* $B_r$, and its only contribution is $J_S$. As the external field goes negative (that is, its amplitude increases with a direction opposite to the magnetic polarization), we move towards the second quadrant of the hysteresis loop. With negative $H$ and unchanged $J$, $B$ reduces and the normal curve drops. At the *coercive field* $H_c$, the net magnetic flux density in the material will be zero, but it still is magnetized! The material will retain its magnetization until it quickly drops to zero as $H$ reaches a given value known as *intrinsic coercive field* $H_{ci}$. This value represents how strongly a material can maintain its magnetization. As $H$ gets stronger past $H_{ci}$, polarization quickly aligns with the external field and the material once again reaches magnetization saturation, but now in the opposite direction from the previous state. If the external applied field goes towards positive values, we have a mirrored behavior and the hysteresis loop is closed.
Please keep in mind that a realistic hysteresis curve will not show a completely flat behavior for magnetic polarization $J$, as in reality its value slightly drops below $J_S$ with decreasing $H$ even before the coercive field value.
### Considerations on magnetic polarization and hysteresis curve
When we deal with the magnetic polarization or the magnetization of a magnet, we are dealing with average values of the material response. If we zoom in inside the magnet, we would see an ensemble of magnetic domains each with its local magnetic polarization. Such localized domains can have polarizations with different directions and intensities. The collective behavior of all such domains is what gives the permanent magnet its characteristics. Think of a big LEGO block built with several smaller LEGO pieces, each piece with its own polarization. The behavior and characteristics of the block are a result from the average of all pieces.

Thus, when we obtain an hysteresis curve, using for example a vibrating sample magnetometer equipment, we are influencing each and every magnetic domain with an homogeneous field $H$ while we measure the collective response $J$. The external field will exert influence over all domains, trying to make the local polarization align with it. Reaching saturation $J_S$ means that the vast majority of domains have parallel local polarizations. The same applies for when $J = 0$. This does not mean that the material lost its magnetic polarization. But the local domains are all oriented along different directions that in average they cancel each other out. So at the macroscopic scale, it is as if the material is no longer magnetized.

## So what are we measuring?
When a magnetic field sensor is placed close to a permanent magnet, its output will reflect the magnetic flux density $B$ acting on it. In other words, it will behave according to the normal curve of the hysteresis loop! If in your application the applied external field can be considered zero, it seems intuitive to just get the remanence $B_r$ value from the magnet's datasheet and use that as the polarization value in your Magpylib simulation. But when comparing the simulation results with experimental data, discrepancies arise. The measured $B$ is smaller, and you already verified that other parameters such as the airgap are correct! So, what can it be?

## Datasheet of a permanent magnet
Let's imagine you are working with a sintered NdFeB permanent magnet from Bomatec (https://www.bomatec.com/en), as an example. You will easily find values of remanence, coercivity, operating temperature, density and others, usually presented by the magnet supplier in a table like this:

![](../../_static/images/gallery_tutorial_magnet_table.png)

After a quick glance, you would use the nominal $B_r$ value in tesla as the magnetic polarization of the Magpylib magnet. After all, we saw in the hysteresis loop that this coincides with $J$ when the field $H$ is zero. But inside the datasheet of the chosen magnet, there is an interesting figure:

![](../../_static/images/gallery_tutorial_magnet_datasheet.png)

This figure shows a focus on the second quadrant of the hysteresis loop, where only negative values for $H$ are present. For two magnet temperatures, the intrinsic and normal curves are shown, being intersected with additional straight lines. To obtain the correct magnetic polarization value $J$ for the simulation, this figure has to be studied.
### When the external applied field is not really external
Consider a permanent magnet surrounded by non-magnetic medium, such as a magnet in air. Outside of it, the only contribution to $B$ measured by a sensor can be $H$, as $J = 0$. The source of this field is the magnetic polarization itself, and as it "leaks" outside the magnet it is also known as the *stray field*. But it also exists inside the magnet itself, in a direction opposite to that of the polarization! Because it acts as if trying to reduce the polarization of the magnet itself, another name for this field is the *demagnetizing field*. The figure below, obtained using Magpylib, (see the streamplot tutorial here[link]), shows the comparison between computed $H$ and $B$ for a rectangular magnet with homogeneous magnetization.
![Alt text](../../_static/images/gallery_tutorial_magnet_fieldcomparison.png)
Thus, with a positive magnetic polarization and negative $H$, datasheets focus on the second quadrant of the hysteresis loop.
### Getting the correct polarization
Then how do we obtain the correct value to be used in Magpylib simulation? For an homogeneous magnetized magnet, this self demagnetizing field depends only on the geometry of the material, through the *permeance coefficient*. The grey straight lines crossing the intrinsic and normal curves represent this coefficient.
![Alt text](../../_static/images/gallery_tutorial_magnet_LDratio.png)
Using a cylindrical magnet with an axial magnetization as an example, the measures of interest are its length $L$ and diameter $D$. Knowing the $L/D$ ratio, we follow the related permeance coefficient line and verify the point where it intersects the normal curve. This point is known as the *working point* of the permanent magnet. The corresponding magnetic flux density $B_W$ is what a sensor would measure if it were inside the magnet. But to get what is the actual magnetic polarization of the magnet, we follow a vertical line straight up from the working point and verify where the intrinsic curve gets intersected. The corresponding magnetic polarization $J_W$ is the value to be used when creating your Magpylib magnet.
Back to the datasheet: according to it, at room temperature the remanence $B_r$ is 1.22 T. If it were a cylinder magnet with $L/D = 1.5$, the polarization value in your Magpylib simulation has to be 1.2 T.
![Alt text](../../_static/images/gallery_tutorial_magnet_datasheet2.png)
The difference in this case is small, but it can be much greater in many others! It depends on the grade, material, shape and quality of the magnet. You may find datasheets in which $J_W$ for the magnet geometry is already indicated, sometimes referred to as "$B_r$ at the working point".

## Example with experimental data
This page will be updated with a comparison between experimental data and Magpylib simulation for different L/D ratios. The permanent magnets were kindly provided by Bomatec.

**Exterior reference**
G. Martinek, S. Ruoho and U. Wyss. (2021).*Magnetic Properties of Permanents Magnets & Measuring Techniques* [White paper]. Arnold Magnetic Technologies. https://www.arnoldmagnetics.com/blog/measuring-permanent-magnets-white-paper/