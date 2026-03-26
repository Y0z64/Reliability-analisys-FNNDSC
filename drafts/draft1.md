# Summary analisys

### Volume

![1774547438563.png](../assets/1774547438563.png)
> 23 subjects in a range of 22-32 weeks GA (inclusive)

- Total volume growth showed a strong linear assosciaton with GA consistent with expected brain growth around this age
- Greater variability is observed on the 3 subjects above 32 weeks GA, this could be explained by a *subpar performance on subplat and cortical plate segmentation by our segementation model since its fitted for subjects under this GA.**
- *: *Subjects >32 weeks GA also have a expected volume decline on the subplate which may be affecting the segmentation and volume measures*

<!-- Insert graph here -->
> Volume differences per tissue against GA with outliers (RANSAC)

![1774551183742.png](../assets/1774551183742.png)
> Dice coefficient of the volume based on tissue and model
> * 5-label model does segment supblate, added for comparision

- Both suplate and cortical plate have acceptable dice scores
- The SP model appears to outperform the 5-label model even on the cortical plate, this could be explained becase the SP model is more adept at processing subjects <32 weeks GA.

![1774551655623.png](../assets/1774551655623.png)
![1774551664734.png](../assets/1774551664734.png)
> Voxel count and native volume across splits

- Voxel count plot shows instability in the measurements over different splits of the same subject and across comparision groups, however across comparision groups the values seem to be relatively with a few outliers
- Variance is reduced when accounting for native scaling as shown in the native volume plot where values are stabilized across splits and comparision groups
- * Outliers fall in the category of very low or very high GA in both measures which could be *explained by biological variability or lower segmentation reliability in under and over a set GA range*

![1774552122050.png](../assets/1774552122050.png)
![1774552103516.png](../assets/1774552103516.png)
> Comparisions should be made in red-blue and green-yellow groups (S1-S2) (S3-S4). Sorted by value.

- Voxel count plot shows groth and generallly low variability across comparision groups. Voxel counts seem to not be completely dependent on GA.
-** Altough variance is observed across comparision groups, it seems to be unrelated to both GA and general volume value. Variability must be explained by another factor.**
- Native volume plot shows expected supralinear growth with increase accelerated on subjects with higher GA.

### Quality metrics
![1774552595058.png](../assets/1774552595058.png)
> SNR is calculated by the following formula:
> ```latex
> SNR = \frac{mean}{\sigma}

- Higher SNR on subplate than cortical plate, *could be explained SP is a relatively homogeneous zone with a more uniform signal intensity than cortical plate*
- Slightly higher difference between S1-S2 than S2-S4 in subplate, most probably not sigmificant.

![1774553312453.png](../assets/1774553312453.png)
![1774553317110.png](../assets/1774553317110.png)
> RANSAC model for |SNR diff| vs |volume diff| for both comparision groups

- Trying to measure the SNR significance correlated by the difference in measured volume difference across comparision groups. Maybe trying to check if SNR is correlated to a higher variance in segmentation, meaning lower reliability?
- *SNR seems significant at least when using volume difference as a quality metric*

##### Correlation matrix
![1774553840416.png](../assets/1774553840416.png)

- GA was the only variable consistenly correlated with absolute volume differences across both intependent splits
- QA and SNR show no consistent association with volume differences across split pairs
- Segmentation reliability is not driven by image quality, based on this quality metrics at least

### CNR
![1774554504225.png](../assets/1774554504225.png)
> CNR against GA
> Measured by using 1mm bands in both zones with a 0.5mm intermediary zone on the side of the subplate tissue