# Modelling marine traffic in the ice-covered Baltic Sea
Prediction of vessel routes, speed, traffic and ETA using grid-based methods and shortest path algorithm. Done as a part of a Master's thesis work. 

**_Summer (red) and winter (blue) vessel tracks in the Baltic Sea:_**
![Summer and winter AIS tracks](https://raw.githubusercontent.com/hakola/marine-traffic-modelling/master/figs/balticsea_ais.png)

### Highlights
* Notebooks 1-6 will guide how the models are created
* AIS observation and dirways from 2017-19 (data/datasets.zip & http://ieee-dataport.org/2099).
    *  Training set containing 14 million AIS observations from the summer and fall of 2017-18
    *  Summer and winter validation sets containing 120 thousand observations from 380 voyages
    *  More data such as port locations and sea areas.   
* Method for calculating voyages out of AIS data (see EXTRA 1. Voyage Mapping)
* Method for extracting mooring areas from AIS data (see EXTRA 3. Extract ports from observation data)

#### Abstract:
> Icebreaking activity and seasonal ice propose challenges for marine traffic prediction in the Baltic Sea. Traffic prediction is a vital part in the planning of icebreaking activities. However, the prediction is still mostly manual task. To combat this, the aim of this thesis is to examine factors influencing marine traffic modelling in ice-covered waters and propose a novel A*-based method for modelling traffic in ice. The current state of the marine traffic modelling and factors affecting vessel movement are concluded by examining the literature and historical vessel tracks.
><br/><br/>
>To summarize the literature review, the field of traffic modelling research is growing and shows great promise. However, the biggest challenges are evaluation of results and the lack of publicly available datasets. Moreover, the current approaches to model vessel movement in ice show promise but they fail to capture how icebreaking activity influences vessel routes.
><br/><br/>
> The proposed model consists of sea, maneuverability, route and speed modelling. The model uses historical AIS data, topography of the sea, vessel type and dirways as main data inputs. The model is trained with summer tracks and dirways are used for modelling the ice channels kept open by icebreakers. The accuracy of the model is evaluated by examining route, speed, traffic and ETA (estimated time of arrival) prediction results separately. Moreover, the area between the actual and predicted route is introduced as an accuracy measure for route prediction. 
><br/><br/>
>The model shows that winter route prediction can be improved by incorporating dirways to the modelling. However, the use of dirways did not affect speed, traffic or ETA prediction accuracy. Finally, the AIS datasets and the source code used in this thesis are published online.


### Approach
* Sea model (Notebooks 1-2)
* Manoeuvrability model (Visualized in Notebook 2)
* Route model (Notebook 4)
* Speed model (Notebook 6)
* Transition probability (Notebook 5)
* A* shortest path algorithm for route prediction (Notebooks 7-8 and pygradu/shortest_path.py)

**_Winter route prediction example:_**
![Winter route prediction example](https://raw.githubusercontent.com/hakola/marine-traffic-modelling/master/figs/winter_route_prediction_example.png)

