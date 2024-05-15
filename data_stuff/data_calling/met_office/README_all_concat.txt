PAMIP data provided by Tido Semmler and Elisa Manzini to Rosie Eade (2020)
tido.semmler@awi.de
elisa.manzini@mpimet.mpg.de
rosie.eade@metoffice.gov.uk

ECHAM6 Medium res T127 from AWI

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/AWI/T127


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/AWI/T127


 100 members:  pdSST-pdSIC  pdSST-futArcSIC
   epfy  epfz  utendepfd  vtem  wtem

  DJF mean (single members): DJF_MEAN/ pdSST-pdSIC  pdSST-futArcSIC
   psl  ta  tas  ua



----------------------------------------------
PAMIP data provided by Rym Msadek to Rosie Eade (2020)
rym.msadek@cerfacs.fr
rosie.eade@metoffice.gov.uk

CNRM-CM6-1 from CERFACS

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/CERFACSnew


Archived to MASS Dec 2022
Monthly Mean Output
moose:/adhoc/projects/pamip/CERFACS

 300 members:  pdSST-pdSIC  pdSST-fuArcSIC
   psl  ta  tas  ua  va  wap
   tem		# EP Fluxes from daily data Primitive Eqn (calc by Rosie using ncl code from Elisa Manzini)
   qg_tem	# EP Fluxes from daily data Q-g Eqn (calc by Rosie using ncl code)

 110 members (r91-200):  pdSST-pdSIC
   hfls  hfss  pr  siconca  uas  vas  zg
 
 100 members: pdSST-futOkhotskSIC pdSST-futBKSeasSIC
   psl ua
 
 100 members: pdSST-piArcSIC
 psl

Daily Mean Output
moose:/adhoc/projects/pamip/CERFACS/daily
 300 members:  pdSST-pdSIC  pdSST-fuArcSIC
   ta ua va



----------------------------------------------
PAMIP data provided by Lantao Sun to Rosie Eade (2019)
ltsun@rams.colostate.edu
rosie.eade@metoffice.gov.uk

CESM2 from NCAR

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/CESM2


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/CESM2

 100 members:  pdSST-pdSIC  pdSST-futArcSIC
   epfy  epfz  psl  ta  tas  ua  utendepfd  vtem  wtem

 +100 members but only as ensemble mean (r101-200):  emean/pdSST-pdSIC/mem101_200  emean/pdSST-futArcSIC/mem101_200
   psl  ta  tas  ua



----------------------------------------------
PAMIP data provided by Michael Sigmond to Rosie Eade (2019)
Michael.Sigmond@canada.ca 
rosie.eade@metoffice.gov.uk

CanESM5 from Canadian Centre for Climate Modelling and Analysis

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/CanESM


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/CanESM5

 100 members:  pdSST-pdSIC  pdSST-futArcSIC
   epfy  epfz  psl  ta  tas  ua  utendepfd  vtem  wtem

 +200 members but only zonal mean (r101-300):  mem101_300/pdSST-pdSIC/  mem101_300/pdSST-futArcSIC/
   epfy  epfz  tazm  uazm  vtem  wtem




----------------------------------------------
PAMIP data provided by Michael Sigmond to Rosie Eade (2019)
Michael.Sigmond@canada.ca 
rosie.eade@metoffice.gov.uk

CanESM5-G from Canadian Centre for Climate Modelling and Analysis
- Same model as CanESM5 except with different settings of orographic gravity wave drag (which determines the basic state).

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/CanESM5-G


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/CanESM5-G

 100 members:  pdSST-pdSIC  pdSST-futArcSIC
   epfy  epfz  tazm  uazm



----------------------------------------------
PAMIP data provided by Yannick Peings to Rosie Eade (2019)
ypeings@uci.edu
rosie.eade@metoffice.gov.uk

E3SM from UCI

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/E3SM


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/E3SM


200 member ensembles:  pdSST-pdSIC  pdSST-futArcSIC
  psl  ta  ua
  epfy  epfz  # EP Fluxes primitive equations on daily data, only for pdSST-pdSIC


100 member ensembles:  PAMIP-1.1-E3SM  PAMIP-1.6-E3SM 
  psl  ta  tas  ua  va
  epfy  epfz # EP Fluxes primitive equations on daily data, only for PAMIP-1.1-E3SM
  mon_tem # EP Fluxes and tem diagnostics, primitive equations on monthly data
  qg_epfy # EP Fluxes qg_epfz (Q-G equations on daily data)

Ensemble mean DJF mean (100 members): emean/DJF_MEAN/ PAMIP-1.1-E3SM  PAMIP-1.6-E3SM 
  wap
  mon_gq_utendepfd # EP Flux divergence Q-G equations on daily data



----------------------------------------------
Data provided by 
Xavier Levine <xavier.levine@bsc.es>
- copied to jasmin by Rosie Eade <rosie.eade@metoffice.gov.uk> 30.01.20

EC-EARTH3 from BSC

Experiments 1.1, 1.6, 3.1, 3.2, 4.1, 4.2

moose:adhoc/projects/pamip/EC-EARTH3
Originally downloaded data to /data/users/hadre/DATA/PAMIP/external/EC-EARTH3

  pdSST-pdSIC & pdSST-futArcSIC (150 members, 1 file for each, monthly means)
    epfy  epfz  psl  ta  tas  ua  va vtrea wap tem
    vtrea = Covariance of V wind and temperature
    tem = tem diagnostics (vtem, wtem)

  pdSST-futBKSeasSIC & pdSST-futOkhotskSIC (150 members)
    psl  ua

  modelSST-pdSIC & modelSST-futArcSIC (175 members)
    psl  ua

Also have ensemble mean output for quasi-geostrophic version of some tem diagnostics
moose:adhoc/projects/pamip/EC-EARTH3
  pdSST-pdSIC & pdSST-futArcSIC
    qg_epfy  qg_epfz  qg_utendepfd


----------------------------------------------
PAMIP data provided by Bian HE to Rosie Eade (2019)
heb@lasg.iap.ac.cn
rosie.eade@metoffice.gov.uk

FGOALS-f3-L from CAS (IAP)

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/FGOALS-f3-L


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/FGOALS-f3-L

 100 member ensembles: pdSST-pdSIC  pdSST-futArcSIC
   psl  ta  tas  ua
   epfy  epfz	# EP Fluxes from daily data but only 8 levels (using Q-G equations?)
   (some variables also available from CMIP6 archive e.g. BADC)
   (va but not all 100 members available)

 Ensemble mean monthly mean: emean/ pdSST-pdSIC  pdSST-futArcSIC
   vtem  wtem		# EP Fluxes from daily data but only 8 levels (using Q-G equations?)
   mon_vtem  mon_wtem	# EP Fluxes from monthly data but 19 levels (using Q-G equations?)





----------------------------------------------
PAMIP data provided by Holly Ayres and Amber Walsh to Rosie Eade (2019/2020)
ha392@exeter.ac.uk
a.l.walsh@exeter.ac.uk
rosie.eade@metoffice.gov.uk

HadGEM3-N96 from University of Exeter

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/HadGEM3_N96


Archived to MASS Dec 2022
Monthly Mean Output
moose:/adhoc/projects/pamip/HadGEM3_N96

westerly QBO and easterly QBO experiments
pdSST-pdSIC
pdSST-futArcSIC

See moose:/adhoc/projects/pamip/HadGEM3_N96/README.txt for full details

----------------------------------------------
PAMIP data provided by Guillaume Gastinaux to Rosie Eade (2020)
ggalod@locean-ipsl.upmc.fr
rosie.eade@metoffice.gov.uk

IPSL from IPSL

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/IPSL


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/IPSL

 200 member ensembles: pdSST-pdSIC  pdSST-futArcSIC
   psl  ta  tas  ua  va  wap
   epfy  epfz  vtem  wtem  utendepfd  psitem # EP Fluxes and tem diagnostics from daily data Primitive Eqn



----------------------------------------------
PAMIP data provided by Masato Mori to Rosie Eade (2019)
masato@atmos.rcast.u-tokyo.ac.jp
rosie.eade@metoffice.gov.uk

MIROC6 from MIROC

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/MIROCnew


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/MIROC6

 100 member ensembles: pdSST-pdSIC  pdSST-futArcSIC
   epfy  epfz  psl  ta  tas  ua  utendepfd  vtem  wtem

 100 member ensembles: pdSST-futBKSeasSIC
   epfy  epfz  psl  tas  utendepfd




----------------------------------------------
PAMIP data provided by Lise Seland to Rosie Eade (2019)
lisesg@met.no
rosie.eade@metoffice.gov.uk

NorESM2 from NCC

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/NorESM2


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/NorESM2

 100 member ensembles: pdSST-pdSIC  pdSST-futArcSIC
   epfy  epfz  psl  ta  tas  ua  vtem  wtem
   (200 members available from CMIP6 archive e.g. BADC)

  DJF mean (single members): DJF_MEAN/ modelSST-pdSIC  modelSST-futArcSIC
   psl ua



----------------------------------------------
Data provided by 
Jan Streffing <Jan.Streffing@awi.de>
- copied to jasmin by Rosie Eade <rosie.eade@metoffice.gov.uk> 12.11.20

OpenIFS from AWI

3 Resolutions: 
High res T1279 (100 members), Medium res T511 (99 members) and Low res T159 (299 members).


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


----------------------------------------------

PAMIP data provided by Yannick Peings to Rosie Eade (2019)
ypeings@uci.edu
rosie.eade@metoffice.gov.uk

SC-WACCM (without QBO) from UCI

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/SC-WACCM-noQBO


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/SC-WACCM-noQBO

 300 member ensembles:  PAMIP-1.1-noQBO  PAMIP-1.6-noQBO
   epfy  epfz  omega  va

  DJF mean (single members): DJF_MEAN/ PAMIP-1.1-noQBO  PAMIP-1.6-noQBO
    psl

  Ensemble mean DJF mean: emean/DJF_MEAN/ PAMIP-1.1-noQBO  PAMIP-1.6-noQBO
    depf  epfy  epfz  omega  ta  tas  ua  va  
    wdf (this is utendepfd)



----------------------------------------------
PAMIP data provided by Yannick Peings to Rosie Eade (2019)
ypeings@uci.edu
rosie.eade@metoffice.gov.uk

SC-WACCM from UCI

Some output remains in original $DATADIR directory
/data/users/hadre/DATA/PAMIP/external/SC-WACCM


Archived to MASS Jan 2023
Monthly Mean Output
moose:/adhoc/projects/pamip/SC-WACCM

 300 member ensembles:  PAMIP-1.1-QBO  PAMIP-1.6-QBO
   ta  tas  va  wap
   uazm  # u-wind zonal mean
   epfy  epfz # EP Fluxes primitive equations on daily data 
   qg_epfy  qg_epfz # EP Fluxes Q-G equations on daily data
   mon_epfy  mon_epfz  mon_tem # EP Fluxes and tem diagnostics, primitive equations on monthly data
   mon_qg_utendepfd # EP Fluxes Q-G equations on monthly data 


  DJF mean (single members): DJF_MEAN/ PAMIP-1.1-QBO  PAMIP-1.6-QBO  PAMIP-3.2-QBO
    psl
    tas for 3.2

  Ensemble mean DJF mean: emean/DJF_MEAN/ PAMIP-1.1-QBO  PAMIP-1.6-QBO   PAMIP-3.2-QBO
    ua
    epfy  epfz  ta  ua  va  depf  wdf for 3.2



----------------------------------------------
Data provided by 
Javier Garcia-Serrano <j.garcia-serrano@meteo.ub.edu>
- copied to jasmin by Rosie Eade <rosie.eade@metoffice.gov.uk> 30.01.20

SPEEDY from Universitat de Barcelona
Experiments 1.1 and 1.6 and 3.2
100 members

ISSUES:
* Sea ice forcing seems wrong as temperature pattern doesn't look as would expect
* Time variable from files - isn't clear which month is which

Monthly Means
moose:adhoc/projects/pamip/SPEEDY

pdSST-pdSIC and pdSST-futArcSIC
psl  ta  tas  ua
epfluxwa epfluxqg  - epflux terms (full with wap and quasi-geostrophic) calculated from monthly means of:
  uv  vT  va  wap

pdSST-futBKSeasSIC/
psl  ta  tas  ua

Originally data in
/data/users/hadre/DATA/PAMIP/external/SPEEDY


----------------------------------------------
