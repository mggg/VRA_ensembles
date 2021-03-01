# VRA_article

This code repository has the Texas and Louisiana elections models, their input data and files. 

## Texas

*TX_elections_model* is the main model file and the *run_functions* file has supporting functions. The user can run a neutral ReCom chain, a constrained (rejection sampling) chain based on district VRA-effectiveness scores, or a constrained chain based on district demographic data. The only parameters for the user to enter are the 'User Input Parameters' in *TX_elections_model.py*. They are described below.

To get started, download *all* input files - including the additional files linked below - into the same local *TX/* directory. Running the *TX_elections_model.py* script (or 'python TX_elections_model.py' in the command line from the *TX/* directory) will make an 'outputs' folder in the same directory.

### Data and Input Files ###

All input files are in this code repository, except for 3 files that were too large to store here. They are the Texas VTD shapefile (with electoral data attached) and two files that have precinct-level EI data. They can be downloaded (and added to you *TX/* folder) [HERE](https://www.dropbox.com/sh/k78n2hyixmv9xdg/AABmZG5ntMbXtX1VKThR7_t8a?dl=0). 

* ***Candidate_Race_Party***: This file has every statewide candidate in each race used in the model. It also has their race and party affiliation.
* ***TX_columns***: Full column names for electoral data used in the shapefile. This file is needed because shapefiles have a 10 character limit.
* ***TX_elections***: Every election's year, type (primary, runoff or general) and "set" (the name used to refer to the set of an election's primary, runoff if applicable and general).
* ***dropped_elecs***: Elections we removed from the analysis because they had uncontested Democratic primaries. The user can add more elections to this file if they want to limit the analysis further.
* ***TX_logit_params***: Intercepts and coefficients for the best fit logit functions for the statewide, district and equal model scores.
* ***recency_weights***: Election weights according to their years.
* ***ingroup_weight***: Election weights according to whether the Latino (Black) prefferred candidate is Latino (Black).
* ***statewide_rxc_EI_preferences***: Statewide EI results. Contains EI-predicted candidates of choice for Black, Latino, White and Other voters for each statewide election. Also listed are point estimates for level of support from these voters and the proportion of EI draws that identified these candidates as the preferred candidates.

* ***TX_VTDs***: The 2018 Texas VTD shapefile from The Texas Legislative Council. We have appended 2010 Census population data, ACS CVAP data for several years and election returns also from the Texas Legislative Council.

* ***mean_prec_vote_counts***: Point estimates for precinct-level vote counts by race for each statewide candidate.  These estimates are derived by averaging precinct-level EI draws.

* ***prec_count_quants***: Quantile (octile) endpoints for precinct-level vote counts by race for each statewide candidate derived from the distribution of precinct-level EI draws.



### User Input Parameters ###
These are the only values in the model the user has to change. They are near the top of the *TX_elections_model* file.

* ***total_steps***: The number of ReCom Markov chain steps in the run. 
* ***pop_tol***: Maximum allowable population deviation from the ideal (total population/number of districts).
* ***run_name***: Name for the chain run (this will appear in the output file names).
* ***start_map***: The first map of the chain. Can enter the enacted Congressional map ('CD'), a seed for a demographic-constrained run ("Seed_Demo") or 'new_seed' to start with a randomly generated plan.
* ***effectiveness_cutoff***: Threshold for counting a district as effective for a particular group. 
* ***ensemble_inclusion***: Set to 'True' to do a constrained run based on VRA-effectiveness scores (the score type is set in the *model_mode* parameter). The inclusion criteria requires a plan to have 8 Latino-effective districts, 4 Black-effective districts and 11 total districts that are effective for one or both groups. A district is Latino (Black) effective if its Latino + Overlap (Black + Overlap) scores exceeds the *effectiveness_cutoff*.
* ***ensemble_inclusion_demo***: Set to 'True' to do a constrained run based on CVAP demographic constraints. The demographic inclusion criteria requires at least 8 districts above 50% HCVAP and at least 4 districts above 25% BCVAP. *ensemble_inclusion* and *ensemble_inclusion_demo* cannot both be True in the same run. Note, if *ensemble_inclusion_demo* is True, the *start_map* must be 'Seed_Demo', as the enacted Congressional map does not meet our demographic thresholds.
* ***record_statewide_modes*** Set to 'True' to compute and record the 'statewide' and 'equal' effectiveness scores. If *ensemble_inclusion* is True and *model_mode* is set to 'statewide' or 'equal', then this setting must be 'True'.
* ***record_district_mode*** Set to 'True' to compute and record the 'district' effectiveness score. If *ensemble_inclusion* is True and *model_mode* is set to 'district', then this setting must be 'True'.
* ***model_mode***: 'statewide', 'equal' or 'district'. For constrained runs, this determines the score the inclusion criteria uses when evaluating a plan.
* ***store_interval***: The number of chain steps between data-storing and resetting.


### Output Files ###
These files will be stored in the 'outputs' folder of your local directory.

* ***store_plans***: Every nth plan is stored in this file, where n is the user-set *store_interval*. The first two columns are VTD IDs and indices, and each subsequent column has their district assignments for a plan.
* ***hisp_prop_df***: Each column is a district, and each row has the HCVAP shares of district CVAP for a plan.
* ***black_prop_df***: Each column is a district, and each row has the BCVAP shares of district CVAP for a plan.
* ***white_prop_df***: Each column is a district, and each row has the WCVAP shares of district CVAP for a plan.
* ***pres16_df***: Each column is a district, and each row has the Democratic district vote shares in the 2016 Presidential general race for a plan.
* ***pres12_df***: Each column is a district, and each row has the Democratic district vote shares in the 2012 Presidential general race for a plan.
* ***sen18_df***: Each column is a district, and each row has the Democratic district vote shares in the 2018 Senate general race for a plan.
* ***gov18_df***: Each column is a district, and each row has the Democratic district vote shares in the 2018 Governor general race for a plan.
* ***centroids_df***: Each column is a district, and each row has the district centroids for a plan.
* ***map_metric***: Has the number of Latino effective districts, Black effective districts and total effective districts accoring to the statewide, equal and district scores for each plan. It also has the number of county splits and cut edges per plan.
* ***final_state_prob_df***: (if *record_statewide_modes* is set to 'True') Each column is a district, each row has the 4-way *statewide* VRA-effectiveness distributions for a plan. The distributions are in the order (L,O,N,Ov) (Latino-effectivenes, Black-effectiveness, Neither-effectiveness, Overlap-effectiveness).
* ***final_equal_prob_df***:  (if *record_statewide_modes* is set to 'True') Each column is a district, each row has the 4-way *equal* (unweighted) VRA-effectiveness distributions for a plan. 
* ***final_dist_prob_df***:  (if *record_district_mode* is set to 'True') Each column is a district, each row has the 4-way *district* VRA-effectiveness distributions for a plan.


## Louisiana

All input files for Louisiana are in this code repository, except for 3 files that were too large to store here. Those are the Louisiana precinct shapefile (with electoral data attached) and two files that have precinct-level EI data, which can be downloaded [HERE](https://www.dropbox.com/sh/urf7qnt0egepxy3/AABsRPmVQJQBvB8P0FztaYH1a?dl=0).

The model's data, inputs and outputs are analogous to those for the Texas model. Major differences include tracking outputs for only one major minority group (Black voters) instead of two. Therefore there are no longer 4-way district level probability distributions in the output files, etc.

Also, the LA model isn't currently set up to do constrained runs based on demographic thresholds. So there is no *ensemble_inclusion_demo* user-parameter, but it can do unconstrained runs and constrained runs based on VRA-effectiveness scores (with the *ensemble_inclusion* parameter).

The LA model can be run starting from its current Congressional map ('CD') or State Senate map ('SEND') (or a randomly generated seed map), but the population deviation (*pop_tol* parameter) will need to be adjusted accordingly.



