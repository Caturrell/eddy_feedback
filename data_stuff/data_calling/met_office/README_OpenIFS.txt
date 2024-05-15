Data provided by 
Jan Streffing <Jan.Streffing@awi.de>
- copied to jasmin by Rosie Eade <rosie.eade@metoffice.gov.uk> 12.11.20

OpenIFS from AWI

3 Resolutions: 
High res T1279 (100 members), Medium res T511 (99 members) and Low res T159 (299 members).

monthly means and members

WARNING:
** qg_tem EP Flux diagnostics - scaling to get to CMIP6 units is still not clear
** tem EP Flux diagnostics - member numbering doesn't match the member numbering of other variables for T1279 and T159 (this tem method of calc more consistent with other pamip model methods as used code from Elisa Manzini)
  => must use zonal mean wind field from tem file if want to correlate members with ep flux members
** Some variables had less than the max number of members, hence T511 uses 99 mem and T159 uses 299 mem. The sub-set of variables that was availabled for these members are in OpenIFS/missing_mem/

moose:/adhoc/projects/pamip/OpenIFS
Originally output to 
/data/users/hadre/DATA/PAMIP/external/OpenIFSnew/

----------------------------------
Low Resolution T159

 moose:/adhoc/projects/pamip/OpenIFS/T159
 299 members: pdSST-futArcSIC  pdSST-pdSIC
 Should be 300 members, but member E600 missing for some variables 
 - so removed non-missing E600 to /data/users/hadre/DATA/PAMIP/external/OpenIFSnew/missing_mem/

   psl  ta  tas  ua  va
   tem 		# EP Fluxes and tem diagnostics using primitive equations
   		# Used Elisa Manzini's code so easy to scale to CMIP6 units
		# tem member numbering doesn't match the member numbering of other variables, so need to use ua-zonal mean within tem file to match with ep fluxes
   qg2_tem 	# EP Fluxes and tem diagnostics using Q-G equations. Downloaded Jan 2021
   		# All 300 members, but don't seem to match those from qg_tem
		# Not sure what scalings need to be applied to convert to CMIP6 units
   qg_tem 	# EP Fluxes and tem diagnostics using Q-G equations. Downloaded Sep 2020
		# Only 85 members and only for pdSST-pdSIC
		# Not sure what scalings need to be applied to convert to CMIP6 units

----------------------------------
Medium Resolution T511

 moose:/adhoc/projects/pamip/OpenIFS/T511
 99 members: pdSST-futArcSIC  pdSST-pdSIC
 Should be 100 members, but member E300 missing for some variables 
 - so removed non-missing E300 to /data/users/hadre/DATA/PAMIP/external/OpenIFSnew/missing_mem/

   psl  ta  tas  ua  va
   tem 		# EP Fluxes and tem diagnostics using primitive equations
   		# Used Elisa Manzini's code so easy to scale to CMIP6 units
		# tem member numbering DOES match the member numbering of other variables, unlike T159 resolution OpenIFS experiments
   qg_tem 	# EP Fluxes and tem diagnostics using Q-G equations. Downloaded Sep 2020
		# Not sure what scalings need to be applied to convert to CMIP6 units



----------------------------------
High Resolution T1279

 moose:/adhoc/projects/pamip/OpenIFS/T1279
 100 members: pdSST-futArcSIC  pdSST-pdSIC

   psl  ta  tas  ua  va
   tem 		# EP Fluxes and tem diagnostics using primitive equations
   		# Used Elisa Manzini's code so easy to scale to CMIP6 units
		# tem member numbering may not match the member numbering of other variables? but
		# Only been given 20 members anyway
   qg_tem 	# EP Fluxes and tem diagnostics using Q-G equations. Downloaded Sep 2020
		# Not sure what scalings need to be applied to convert to CMIP6 units

------------------------------------
