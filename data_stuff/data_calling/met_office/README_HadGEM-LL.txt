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


__Original experiments given in pp format (westerly QBO)__
/data/users/hadre/DATA/PAMIP/external/HadGEM3_N96/
pdSST-pdSIC
pdSST-futArcSIC

100 members (98 for JAS season - see ISSUES below)

variables available are:
apm/*a.pm*.pp
 tas, psl
 ta, ua
- has most of the variables needed for PAMIP analysis (note that ua and ta both on uv grid)
- need to use H when reading in pressure level fields to identify points where pressure level goes below the surface
- pressure level fields look very odd at level nearest surface

aph/*a.ph*.nc
 va (wap?)
 H (heavyside function)

apf/*a.pf*.nc
 vtem, epfy, epfz,
- made by Rosie from daily mean data in *a.pg*

apk/*a.pk*.nc
 wtem, utendepfd, 
- made by Rosie from 10-day mean data in *a.pg*

(apg/*a.pg.nc daily and 10-daily mean data)
(apg/*a.p5.pp monthly mean stash 30312, 30313)
-----------------------------------------------------------------

__Opposite QBO experiments given in nc format (easterly QBO)__
/data/users/hadre/DATA/PAMIP/external/HadGEM3_N96/
pdSST-pdSIC-QBOE
pdSST-futArcSIC-QBOE

95 members (should be 100 members but some data missing/corrupt means remove r001,008,010,011,012 from analysis)

variables available are:
 vtem, wtem, epfy, epfz, utendepfd, tas, psl, ta, ua (not va)

apm/*a.pm*.nc
 tas, psl
 ta, ua
- has most of the variables needed for PAMIP analysis (note that ua and ta both on uv grid)
- use ta when reading in pressure level fields to identify points where pressure level goes below the surface (missing data values of "0" easier to identify in ta than ua fields)
- pressure level fields look very odd at level nearest surface

apf/*a.pf*.nc
 vtem, epfy, epfz,
- made by Rosie from daily mean data in *a.pg*

apk/*a.pk*.nc
 wtem, utendepfd, 
- made by Rosie from 10-day mean data in *a.pg*
-----------------------------------------------------------------



***ISSUES***
*Corrupted files*:
pdSST-pdSIC
  r002i1p1f1/bj524a.pg20000701.nc # JAS
- Substituted with consecutive member for now (1->2) *Didn't remove as DJF still usable, but should skip if using JAS
pdSST-futArcSIC
  r001i1p1f1/bm516a.pg20000701.nc # JAS
- Substituted with consecutive member for now (2->1) *Didn't remove as DJF still usable, but should skip if using JAS

pdSST-pdSIC-QBOE-nc
  r092i1p1f1/br363a.pm2000aug.nc - has no lat/lon values (all == zero)
- overwrite with lat/lon info from earlier member when reading into iris cube.

epflux/pdSST-pdSIC-QBOE-nc
  r094i1p1f1/br363a.ph20000701.nc
- has 5 dates instead of the expected 3? used xconv to get rid of 2 of them, but may still not be correct.

*Missing files*:
epflux/pdSST-futArcSIC-QBOE-nc
  r001i1p1f1/bs720a.ph20010101.nc  <-bs720a.pg20010101.nc
- remove r001 from analysis as this only contains JF of DJF?
  r008,010,011,012 missing 2001 files
- remove r008,010,011,012 from analysis => total of 95 members instead of 100

pdSST-futArcSIC-QBOE-nc
  r068i1p1f1/*2000apr.nc
pdSST-futArcSIC-QBOE-nc
  r068i1p1f1/*2000may.nc
- Substituted with r067 files to simplify loading cube in iris as don't actually need these files anyway

Some members have 2001 June and July, others don't. Removed these to avoid iris cube issues


----------------------------------------
*Calculating of monthly means from daily means*
a.pg
- has daily uep and nep --> calculate monthly means as a.p5 (30312, 30313) using IDL_FILES/calc_apm_pamip_output_plev_daily.pro?
- daily mean vstarbar, uep, nep
  To get vtem (vstarbar), epfz (uep), epfy (nep), calculate monthly means from 1-day means (-> a.ph).
- 10 daily mean wstarbar, epdiv, merid heat flux, u after timestep, v after timestep
  To get utendepfd (epdiv), wtem (wstarbar), calculate monthly means from 10-day means (-> a.pi).
  re-extracted apg from Amber (Aug 2020)
  $DATADIR/CODE/PAMIP/SH_FILES/run_cdo_monmean_select_varlist.com
a.pe has daily H on p lev/uv grid 30301 and ua, va, ta (30201, 30202, 30204)
- need to use H when reading in pressure level fields to identify points where pressure level goes below the surface
----------------------------------------



----------------------------------------
Notes on how downloaded originally

Nov 2019
Originally extracted data from jasmin to 
/scratch/hadre/PAMIP/HadGEM3_N96/
using
/data/users/hadre/DATA/PAMIP/external/HadGEM3_N96/extract_holly.com &
/data/users/hadre/DATA/PAMIP/external/HadGEM3_N96/extract_amber.com &
/data/users/hadre/DATA/PAMIP/external/HadGEM3_N96/extract_amber2.com &
e.g.
rsync -r reade@jasmin-xfer1.ceda.ac.uk:/group_workspaces/jasmin2/realproj/users/hayres/$exp_dir/r$dec$yr\i1p1f1/*/*a.pg*.pp /scratch/hadre/PAMIP/HadGEM3_N96/$exp_dir/r$dec$yr\i1p1f1/
ssh -A reade@jasmin-login1.ceda.ac.uk
ssh -A reade@jasmin-sci1.ceda.ac.uk 
cd /group_workspaces/jasmin2/realproj/users/

Mar 2020
/scratch/hadre/PAMIP/HadGEM3_N96/*/*/*a.pg*.nc
/data/users/hadre/DATA/PAMIP/external/HadGEM3_N96/extract_amber.com &
--> /data/users/hadre/DATA/PAMIP/external/HadGEM3_N96/
PAMIP_pdsst_pdsic/r*/*a.pg*.nc
PAMIP_pdsst_futArc/r*/*a.pg*.nc

rsync -r reade@jasmin-xfer1.ceda.ac.uk:/group_workspaces/jasmin2/realproj/users/alwalsh/PAMIP/$exp_dir/r$dec$yr\i1p1f1/*a.pg*.nc /scratch/hadre/PAMIP/HadGEM3_N96/$exp_dir/r$dec$yr\i1p1f1/

Aug 2020
Downloaded opposite QBO experiments

Apr 2021
ssh -A -Y reade@login1.jasmin.ac.uk
ssh -A -Y reade@sci1.jasmin.ac.uk
ssh -A -Y xfer1.jasmin.ac.uk
ssh -A -Y xfer2.jasmin.ac.uk
ssh -A -Y hpxfer1.jasmin.ac.uk
ssh -A -Y hpxfer2.jasmin.ac.uk

/gws/nopw/j04/realproj/users/reade

(used to be /group_workspaces/jasmin2/realproj/users/reade)
------------



----------------------------------------------
