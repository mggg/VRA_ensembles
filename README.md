# VRA_article

### User Inputs ###

* ***total_steps***: The number of ReCom Markov chain steps in the run. This is the total number of accepted plans.
* ***pop_tol***: Maximum allowable population deviation from the ideal (total population/number of districts).
* ***run_name***: Name for the chain run (this will appear in the output file names).
* ***start_map***: The first map of the chain. Can enter an enacted Congressional or state legislative map ('CD', 'sldl358', 'sldu172', 'sldl309') or 'new_seed' to start with a randomly generated plan.
* ***effectiveness_cutoff***: threshold for counting a district as effective for a particular group 
* ***ensemble_inclusion***: Set to 'True' to do a constrained run based on the VRA-effectiveness score (the score type is set in the *model_mode* parameter). The inclusion criteria requires a plan to have 8 Latino-effective districts, 4 Black-effective districts and 11 total districts that are effective for one or both groups. A district is Latino (Black) effective if its Latino + Overlap (Black + Overlap) scores exceeds the *effectiveness_cutoff*.
* ***ensemble_inclusion_demo***: Set to 'True' to do a constrained run based on CVAP demographic constraints. The demographic inclusion criteria requires at least 8 districts above 50% HCVAP and at least 4 districts above 25% BCVAP. *ensemble_inclusion* and *ensemble_inclusion_demo* cannot both be True in the same run.
* ***model_mode***: 'statewide', 'equal' or 'district'. For constrained runs, this determines the score the inclusion criteria uses when evaluating a plan.
* ***store_interval***: The number of chain steps between storing data.

### Output Files ###

* ***store_plans***: Every 500th plan is stored in this file. The first two columns are VTD IDs and indices, and each subsequent column has their district assignments for a plan.
* ***hisp_prop_df***: Each column is a district, and each row has the HCVAP shares of district CVAP for a plan.
* ***black_prop_df***: Each column is a district, and each row has the BCVAP shares of district CVAP for a plan.
* ***white_prop_df***: Each column is a district, and each row has the WCVAP shares of district CVAP for a plan.
* ***pres16_df***: Each column is a district, and each row has the Democratic district vote shares in the 2016 Presidential general race for a plan.
* ***pres12_df***: Each column is a district, and each row has the Democratic district vote shares in the 2012 Presidential general race for a plan.
* ***sen18_df***: Each column is a district, and each row has the Democratic district vote shares in the 2018 Senate general race for a plan.
* ***gov18_df***: Each column is a district, and each row has the Democratic district vote shares in the 2018 Governor general race for a plan.
* ***centroids_df***: Each column is a district, and each row has the district centroids for a plan.
* ***map_metric***: Has the number of Latino effective districts, Black effective districts and total effective districts accoring to the statewide, equal and district scores for each plan. It also has the number of county splits and cut edges per plan.
* ***final_state_prob_df***: Each column is a district, each row has the 4-way *statewide* VRA-effectiveness distributions for a plan. The distributions are in the order (L,O,N,Ov) (Latino-effectivenes, Black-effectiveness, Neither-effectiveness, Overlap-effectiveness).
* ***final_dist_prob_df***: Each column is a district, each row has the 4-way *district* VRA-effectiveness distributions for a plan.
* ***final_equal_prob_df***: Each column is a district, each row has the 4-way *equal* (unweighted) VRA-effectiveness distributions for a plan. 

