# VRA_article

### User Inputs ###

* total_steps: The number of ReCom Markov chain steps in the run. This is the total number of accepted plans.
* pop_tol: Maximum allowable population deviation from the ideal (total population/number of districts).
* run_name: Name for the chain run (this will appear in the output file names).
* start_map: The first map of the chain. Can enter an enacted Congressional or state legislative map ('CD', 'sldl358', 'sldu172', 'sldl309') or 'new_seed' to start with a randomly generated plan.
* effectiveness_cutoff: threshold for counting a district as effective for a particular group 
* ensemble_inclusion: Set to 'True' to do a constrained run based on the VRA-effectiveness score (the score type is set in the *model_mode* parameter). The inclusion criteria requires a plan to have 8 Latino-effective districts, 4 Black-effective districts and 11 total districts that are effective for one or both groups. A district is Latino (Black) effective if its Latino + Overlap (Black + Overlap) scores exceeds the *effectiveness_cutoff*.
* ensemble_inclusion_demo: Set to 'True' to do a constrained run based on CVAP demographic constraints. The demographic inclusion criteria requires at least 8 districts above 50% HCVAP and at least 4 districts above 25% BCVAP. *ensemble_inclusion* and *ensemble_inclusion_demo* cannot both be True in the same run.
* model_mode: 'statewide', 'equal' or 'district'. For constrained runs, this determines the score the inclusion criteria uses when evaluating a plan.
* store_interval: The number of chain steps between storing data.

