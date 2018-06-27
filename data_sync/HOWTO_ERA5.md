
| Short Name  | Long Name           | Description                                       |  Unit      | ID   |  Aditional Information  |
| ----------- |:--------------------|:--------------------------------------------------|:-----------|:-----|:------------------------|
| PTYPE       | Precipitation type  | Describes the type of precipitation at the surface at the validity time. A precipitation type is assigned wherever there is a non-zero value of precipitation in the model output field (however small). The precipitation type should be used together with the precipitation rate to provide for example indication of potential freezing rain events. Precipitation type (0-8) uses WMO Code Table 4.201 <br> <br> Values of ptype defined in the IFS:<br> <br> 0 = No precipitation<br> 1 = Rain<br> 3 = Freezing rain (i.e. supercooled)<br> 5 = Snow<br> 6 = Wet snow (i.e. starting to melt)<br> 7 = Mixture of rain and snow<br> 8 = Ice pellets<br><br> | (0-8)  |  260015  | GRIB2
|TP |	Total precipitation	| Convective precipitation + stratiform precipitation (CP +LSP). Accumulated field.	| m	| 228 | GRIB2 |

 
