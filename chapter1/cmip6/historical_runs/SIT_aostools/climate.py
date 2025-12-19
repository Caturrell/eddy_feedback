#!/usr/bin/python
# Filename: climate.py
#
# Code by Martin Jucker, distributed under an GPLv3 License
#
# This file provides helper functions that can be useful as pre-viz-processing of files and data

############################################################################################
#
from __future__ import print_function
import numpy as np
import xrft
import xarray as xr
import pdb
import logging
# from numba import jit


#=======================================================================================
## Compute EP flux vectors and divergence
#=======================================================================================


def ComputeEPfluxDivXr(u,v,t,lon='infer',lat='infer',pres='infer',time='time',ref='mean',w=None,do_ubar=False,wave=0):
	""" Compute the EP-flux vectors and divergence terms.

		The vectors are normalized to be plotted in cartesian (linear)
		coordinates, i.e. do not incluxde the geometric factor a*cos\phi.
		Thus, ep1 is in [m2/s2], and ep2 in [hPa*m/s2].
		The divergence is in units of m/s/day, and therefore represents
		the deceleration of the zonal wind. This is actually the quantity
		1/(acos\phi)*div(F).

	INPUTS:
	  u    - zonal wind, xarray.DataArray [m/s]
	  v    - meridional wind, xarray.DataArray [m/s]
	  t    - temperature, xarray.DataArray [K]
	  lon  - name of longitude
	  lat  - name of latitude
	  pres - name of pressure coordinate [hPa/s]
	  time - name of time coordinate (used for ComputeVertEddy)
	  ref  - method to use for dTheta/dp, used for ComputeVertEddy
	  w    - pressure velocity, if not None, xarray.DataArray [hPa/s]
	  do_ubar - compute shear and vorticity correction?
	  wave - only include this wave number. total if == 0, all waves if <0, sum over waves if a list. optional
	OUTPUTS (all xarray.DataArray):
	  ep1  - meridional EP-flux component, scaled to plot in cartesian [m2/s2]
	  ep2  - vertical   EP-flux component, scaled to plot in cartesian [hPa*m/s2]
	  div1 - horizontal EP-flux divergence, divided by acos\phi [m/s/d]
	  div2 - horizontal EP-flux divergence , divided by acos\phi [m/s/d]
	"""
	# some constants
	from .constants import Rd,cp,kappa,p0,Omega,a0
	# shape
	dim_names = FindCoordNames(u)
	if lon == 'infer':
		lon = dim_names['lon']
	if lat == 'infer':
		lat = dim_names['lat']
	if pres == 'infer':
		pres = dim_names['pres']
	initial_order = u.dims
	# geometry
	coslat = np.cos(np.deg2rad(u[lat]))
	sinlat = np.sin(np.deg2rad(u[lat]))
	R      = 1./(a0*coslat)
	f      = 2*Omega*sinlat
	pp0    = (p0/u[pres])**kappa
	#
	# absolute vorticity
	if do_ubar:
		ubar = u.mean(lon)
		fhat = R*np.rad2deg((ubar*coslat)).differentiate(lat,edge_order=2)
	else:
		fhat = 0.
	fhat = f - fhat # [1/s]
	#
	## compute thickness weighted heat flux [m.hPa/s]
	vbar,vertEddy, dthdp_bar = ComputeVertEddyXr(v,t,pres,p0,lon,time,ref,wave) # vertEddy = bar(v'Th'/(dTh_bar/dp))
	#
	## get zonal anomalies
	if isinstance(wave,list):
		# upvp = GetWavesXr(u,v,dim=lon,wave=-1).sel(k=wave).sum('k')
		upvp = GetWavesXrft(u, v, dim=lon, wave=wave)
	elif wave == 0:
		u = u - u.mean(lon)
		v = v - v.mean(lon)
		upvp = (u*v).mean(lon)
	else:
		upvp = GetWavesXr(u,v,dim=lon,wave=wave)
	#
	## compute the horizontal component
	ep1_cart = -upvp
	
	if do_ubar:
		shear = ubar.differentiate(pres,edge_order=2) # [m/s.hPa]
		ep1_cart = ep1_cart + shear*vertEddy # [m2/s2 + m/s.hPa*m.hPa/s] = [m2/s2]
	else:
		shear = 0.
	#
	## compute vertical component of EP flux.
	## at first, keep it in Cartesian coordinates, ie ep2_cart = f [v'theta'] / [theta]_p + ...
	#
	ep2_cart = fhat*vertEddy # [1/s*m.hPa/s] = [m.hPa/s2]
	if w is not None:
		if isinstance(wave,list):
			# w = GetWavesXr(u,w,dim=lon,wave=-1).sel(k=wave).sum('k')
			w = GetWavesXrft(u, w, dim=lon, wave=wave)
		elif wave == 0:
			w = w - w.mean(lon) # w = w' [hPa/s]
			w = (w*u).mean(lon) # w = bar(u'w') [m.hPa/s2]
		else:
			w = GetWavesXr(u,w,dim=lon,wave=wave)
		ep2_cart = ep2_cart - w # [m.hPa/s2]
	#
	#
	# We now have to make sure we get the geometric terms right
	# With our definition,
	#  div1 = 1/(a.cosphi)*d/dphi[a*cosphi*ep1_cart*cosphi],
	#    where a*cosphi comes from using cartesian, and cosphi from the derivative
	# With some algebra, we get
	#  div1 = cosphi d/d phi[ep1_cart] - 2 sinphi*ep1_cart
	div1 = coslat*(np.rad2deg(ep1_cart).differentiate(lat,edge_order=2)) \
			- 2*sinlat*ep1_cart
	# Now, we want acceleration, which is div(F)/a.cosphi [m/s2]
	div1 = R*div1 # [m/s2]
	#
	# Similarly, we want acceleration = 1/a.coshpi*a.cosphi*d/dp[ep2_cart] [m/s2]
	div2 = ep2_cart.differentiate(pres,edge_order=2) # [m/s2]
	#
	# convert to m/s/day
	div1 = div1*86400
	div2 = div2*86400
	#
	# make sure order is the same as input
	new_order = [d for d in initial_order if d != lon]
	if not isinstance(wave,list) and wave < 0:
		new_order = ['k'] + new_order
	# give the DataArrays their names
	ep1_cart.name = 'ep1'
	ep2_cart.name = 'ep2'
	div1.name = 'div1'
	div2.name = 'div2'
	return ep1_cart.transpose(*new_order),ep2_cart.transpose(*new_order),div1.transpose(*new_order),div2.transpose(*new_order), dthdp_bar

#######################################################
def FindCoordNames(ds):
	'''Find the actual dimension names in xr.Dataset or xr.DataArray
	    which correspond to longitude, latitude, pressure

	   INPUTS:
	   	  ds:        xarray.Dataset or DataArray
	   OUTPUTS:
	      dim_names: Dictionary of dimension names.
		  		      Longitude = ds[dim_names['lon']]
					  Latitude  = ds[dim_names['lat']]
					  Pressure  = ds[dim_names['pres']]
	'''
	odims = list(ds.coords)
	ldims = [d.lower() for d in odims]
	dim_names = {}
	# check for longitude
	for lon in ['longitude','lon','xt_ocean','lon_sub1']:
		if lon in ldims:
			indx = ldims.index(lon)
			dim_names['lon'] = odims[indx]
	for lat in ['latitude','lat','yt_ocean','lat_sub1']:
		if lat in ldims:
			indx = ldims.index(lat)
			dim_names['lat'] = odims[indx]
	for plev in ['level','pres','pfull','lev','plev','pressure_level','lev_p']:
		if plev in ldims:
			indx = ldims.index(plev)
			dim_names['pres'] = odims[indx]
	return dim_names

def ComputeVertEddyXr(v,t,p='level',p0=1e3,lon='lon',time='time',ref='mean',wave=0):
	""" Computes the vertical eddy components of the residual circulation,
		bar(v'Theta'/Theta_p).
		Output units are [v_bar] = [v], [t_bar] = [v*p]

		INPUTS:
			v    - meridional wind, xr.DataArray
			t    - temperature, xr.DataArray
			p    - name of pressure
			p0   - reference pressure for potential temperature
			lon  - name of longitude
			time - name of time field in t
			ref  - how to treat dTheta/dp:
			       - 'rolling-X' : centered rolling mean over X days
			       - 'mean'	     : full time mean
                               - 'instant'   : no time operation
			wave - wave number: if == 0, return total. else passed to GetWavesXr()
		OUPUTS:
			v_bar - zonal mean meridional wind [v]
			t_bar - zonal mean vertical eddy component <v'Theta'/Theta_p> [v*p]
	"""
	#
	# some constants
	from .constants import kappa
	#
	# pressure quantitites
	pp0 = (p0/t[p])**kappa
	# convert to potential temperature
	t = t*pp0 # t = theta
	# zonal means
	v_bar = v.mean(lon)
	t_bar = t.mean(lon) # t_bar = theta_bar
	# prepare pressure derivative
	dthdp = t_bar.differentiate(p,edge_order=2) # dthdp = d(theta_bar)/dp
	dthdp = dthdp.where(dthdp != 0)
	# time mean of d(theta_bar)/dp
	print('clipping small values of dthdp to prevent large values of 1./dthdp')
	dthdp = dthdp.where(np.abs(dthdp)>0.02)
	dthdp = dthdp.where(dthdp<0.0)
	# import pdb
	# import matplotlib.pyplot as plt
	# import xarray as xar
	# for t_tick in range(30):
	# 	plt.close('all')
	# 	fig,axes = plt.subplots(1,2)
	# 	xar.plot.hist(1./dthdp[t_tick,...], bins=100, ax=axes[0])
	# 	(1./dthdp[t_tick]).plot.contourf(levels=30, ax=axes[1], yincrease=False)
	# 	plt.savefig(f'hist_{t_tick}.pdf')
	# pdb.set_trace()
	if time in dthdp.dims:
		if 'rolling' in ref:
			r = int(ref.split('-')[-1])
			dthdp = dthdp.rolling(dim={time:r},min_periods=1,center=True).mean()
		elif ref == 'mean':
			dthdp = dthdp.mean(time)
		elif ref == 'instant':
			dthdp = dthdp
	# now get wave component
	if isinstance(wave,list):
		vpTp = GetWavesXrft(v, t, dim=lon, wave=wave)
	elif wave == 0:
		vpTp = (v - v_bar)*(t - t_bar)
		vpTp = vpTp.mean(lon)  # vpTp = bar(v'Th')
	else:
		vpTp = GetWavesXr(v,t,dim=lon,wave=wave) # vpTp = bar(v'Th'_{k=wave})
	t_bar = vpTp/dthdp # t_bar = bar(v'Th')/(dTh_bar/dp)
	#
	return v_bar,t_bar, dthdp

##############################################################################################


def GetWavesXrft(x, y, wave=-1, dim='lon', anomaly=None):

	ftx = xrft.fft(x, dim='lon')
	fty = xrft.fft(y, dim='lon')

	ftx_wavenumbers = ftx.freq_lon.values*360.
	fty_wavenumbers = fty.freq_lon.values*360.

	ftx_mask = xr.zeros_like(ftx.coords['freq_lon'])

	if type(wave)==list:
		for wave_val in wave:
			where_wave = np.where(np.abs(ftx_wavenumbers)==wave_val)[0]
			if len(where_wave)<1:
				logging.info(f'No exact found for wave={wave_val}. Trying alternate method')
				where_wave = np.where(np.abs(np.abs(ftx_wavenumbers) - wave_val)<0.1)
				if np.shape(where_wave)[0]>0:
					logging.info(f'Approx match found at {ftx_wavenumbers[where_wave]}')				
				else:
					pdb.set_trace()
			ftx_mask[where_wave] = 1.0
	elif wave==-1:
		ftx_mask += 1.
	else:
		raise NotImplementedError('Have not implemented alternatives for wave type')

	ftx = ftx * ftx_mask
	fty = fty * ftx_mask

	filter_x = np.real(xrft.ifft(ftx, dim='freq_lon', lag=0))
	filter_y = np.real(xrft.ifft(fty, dim='freq_lon', lag=0))

	prod = filter_x*filter_y

	return prod.mean('lon')

#######################################################

def GetWavesXr(x,y=None,wave=-1,dim='infer',anomaly=None):
	"""Get Fourier mode decomposition of x, or <x*y>, where <.> is zonal mean.

		If y!=None, returns Fourier mode contributions (amplitudes) to co-spectrum zonal mean of x*y. Dimension along which Fourier is performed is either gone (wave>=0) or has len(axis)/2+1 due to Fourier symmetry for real signals (wave<0).

		If y=None and wave>=0, returns real space contribution of given wave mode. Output has same shape as input.
		If y=None and wave<0, returns real space contributions for all waves. Output has additional first dimension corresponding to each wave.

	INPUTS:
		x	   - the array to decompose. xr.DataArray
		y	   - second array if wanted. xr.DataArray
		wave	   - which mode to extract. all if <0
		dim	   - along which dimension of x (and y) to decompose. Defaults to longitude if 'infer'.
		anomaly	   - if not None, name of dimension along which to compute anomaly first.
	OUTPUTS:
		xym	   - data. xr.DataArray
	"""
	from xarray import DataArray
	if dim == 'infer':
		dim_names = FindCoordNames(x)
		dim = dim_names['lon']
	if anomaly is not None:
		x = x - x.mean(anomaly)
		if y is not None:
			y = y - y.mean(anomaly)
	sdims = [d for d in x.dims if d != dim]
	if len(sdims) == 0:
		xstack = x.expand_dims('stacked',axis=-1)
	else:
		xstack = x.stack(stacked=sdims)
	if y is None:
		ystack=None
	else:
		if len(sdims) == 0:
			ystack = y.expand_dims('stacked',axis=-1)
		else:
			ystack = y.stack(stacked=sdims)

	gw = GetWaves(xstack,ystack,wave=wave,axis=xstack.get_axis_num(dim))

	if y is None and wave >= 0: # result in real space
		if len(sdims) == 0:
			stackcoords = x.coords
		else:
			stackcoords = [xstack[d] for d in xstack.dims]
	elif y is None and wave < 0: # additional first dimension of wave number
		stackcoords = [('k',np.arange(gw.shape[0]))]
		if len(sdims) == 0:
			stackcoords = stackcoords + [x[d] for d in x.dims]
		else:
			stackcoords = stackcoords + [xstack[d] for d in xstack.dims]
	elif y is not None and wave >= 0: # original dimension is gone
		stackcoords = [xstack.stacked]
	elif y is not None and wave < 0: # additional dimension of wavenumber
		stackcoords = [('k',np.arange(gw.shape[0])), xstack.stacked]
	gwx = DataArray(gw,coords=stackcoords)
	return gwx.unstack()

def GetWaves(x,y=None,wave=-1,axis=-1,do_anomaly=False):
	"""Get Fourier mode decomposition of x, or <x*y>, where <.> is zonal mean.

		If y!=[], returns Fourier mode contributions (amplitudes) to co-spectrum zonal mean of x*y. Shape is same as input, except axis which is len(axis)/2+1 due to Fourier symmetry for real signals.

		If y=[] and wave>=0, returns real space contribution of given wave mode. Output has same shape as input.
		If y=[] and wave=-1, returns real space contributions for all waves. Output has additional first dimension corresponding to each wave.

	INPUTS:
		x	   - the array to decompose
		y	   - second array if wanted
		wave	   - which mode to extract. all if <0
		axis	   - along which axis of x (and y) to decompose
		do_anomaly - decompose from anomalies or full data
	OUTPUTS:
		xym	   - data in Fourier space
	"""
	initShape = x.shape
	x = AxRoll(x,axis)
	if y is not None:
		y = AxRoll(y,axis)
	# compute anomalies
	if do_anomaly:
		x = GetAnomaly(x,0)
		if y is not None:
			y = GetAnomaly(y,0)
	# Fourier decompose
	x = np.fft.fft(x,axis=0)
	nmodes = x.shape[0]//2+1
	if wave < 0:
			if y is not None:
				xym = np.zeros((nmodes,)+x.shape[1:])
			else:
				xym = np.zeros((nmodes,)+initShape)
	else:
		xym = np.zeros(initShape[:-1])
	if y is not None:
			y = np.fft.fft(y,axis=0)
			# Take out the waves
			nl  = x.shape[0]**2
			xyf  = np.real(x*y.conj())/nl
			# due to symmetric spectrum, there's a factor of 2, but not for wave-0
			mask = np.zeros_like(xyf)
			if wave < 0:
				for m in range(xym.shape[0]):
					mask[m,:] = 1
					mask[-m,:]= 1
					xym[m,:] = np.sum(xyf*mask,axis=0)
					mask[:] = 0
				# wavenumber 0 is total of all waves
                                #  this makes more sense than the product of the zonal means
				xym[0,:] = np.nansum(xym[1:,:],axis=0)
				xym = AxRoll(xym,axis,invert=True)
			else:
				xym = xyf[wave,:]
				if wave >= 0:
					xym = xym + xyf[-wave,:]
	else:
			mask = np.zeros_like(x)
			if wave >= 0:
				mask[wave,:] = 1
				mask[-wave,:]= 1 # symmetric spectrum for real signals
				xym = np.real(np.fft.ifft(x*mask,axis=0))
				xym = AxRoll(xym,axis,invert=True)
			else:
				for m in range(xym.shape[0]):
					mask[m,:] = 1
					mask[-m,:]= 1 # symmetric spectrum for real signals
					fourTmp = np.real(np.fft.ifft(x*mask,axis=0))
					xym[m,:] = AxRoll(fourTmp,axis,invert=True)
					mask[:] = 0
	return np.squeeze(xym)

##helper functions
def GetAnomaly(x,axis=-1):
	"""Computes the anomaly of array x along dimension axis.

	INPUTS:
	  x    - array to compute anomalies from
	  axis - axis along dimension for anomalies
	OUTPUTS:
	  x    - anomalous array
	"""    #bring axis to the front
	xt= AxRoll(x,axis)
	#compute anomalies
	xt = xt - xt.mean(axis=0)[np.newaxis,:]
	#bring axis back to where it was
	x = AxRoll(xt,axis,invert=True)
	return x

# helper function: re-arrange array dimensions
def AxRoll(x,ax,invert=False):
	"""Re-arrange array x so that axis 'ax' is first dimension.
		Undo this if invert=True
	"""
	if ax < 0:
		n = len(x.shape) + ax
	else:
		n = ax
	#
	if invert is False:
		y = np.rollaxis(x,n,0)
	else:
		y = np.rollaxis(x,0,n+1)
	return y

##############################################################################################



#=======================================================================================
## Compute Vstar
#=======================================================================================


def ComputeVstar(data, temp='temp', vcomp='vcomp', pfull='pfull', lon='lon', lat='lat', time='time', ref='mean', wave=-1, p0=1e3):
	"""Computes the residual meridional wind v* (as a function of time).

		INPUTS:
			data  - filename of input file, relative to wkdir, or dictionary with {T,v,pfull}
			temp  - name of temperature field in data
			vcomp - name of meridional velocity field in data
			pfull - name of pressure in inFile [hPa]
			wave  - decompose into given wave number contribution if wave>=0
			p0    - pressure basis to compute potential temperature [hPa]
		OUTPUTS:
			vstar	    - residual meridional wind, as a function of time
	"""
	import netCDF4 as nc

	a0    = 6371000
	g     = 9.81

	# read input file
	if isinstance(data,str):
		print('Reading data')
		update_progress(0)
		#
		inFile = nc.Dataset(data, 'r')
		t = inFile.variables[temp][:]
		update_progress(.45)
		v = inFile.variables[vcomp][:]
		update_progress(.90)
		p = inFile.variables[pfull][:]
		update_progress(1)
		inFile.close()
		#
		v_bar,t_bar = ComputeVertEddy(v,t,p,p0,wave=wave)
		# t_bar = bar(v'Th'/(dTh_bar/dp))
		#		
		dp  = np.gradient(p)[np.newaxis,:,np.newaxis]
		vstar = v_bar - np.gradient(t_bar,edge_order=2)[1]/dp				
	elif type(data)==dict:
		p = data[pfull]
		v_bar,t_bar = ComputeVertEddy(data[vcomp],data[temp],p,p0,wave=wave)
		# t_bar = bar(v'Th'/(dTh_bar/dp))
		#		
		dp  = np.gradient(p)[np.newaxis,:,np.newaxis]
		vstar = v_bar - np.gradient(t_bar,edge_order=2)[1]/dp		
	else:
		v_bar,t_bar, dtdp_bar = ComputeVertEddyXr(data[vcomp], data[temp], pfull, p0, lon, time, ref)
		vstar = v_bar - t_bar.differentiate(pfull,edge_order=2)	

	return vstar

## helper functions
def update_progress(progress,barLength=10,info=None):
	import sys
	status = ""
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	if progress >= 1:
		progress = 1
		status = "\r" #"\r\n"
	#status = "Done...\r\n"
	block = int(round(barLength*progress))
	if info is not None:
		text = '\r'+info+': '
	else:
		text = '\r'
	if progress == 1:
		if info is not None:
			text = "\r{0}	{1}	{2}".format(" "*(len(info)+1)," "*barLength,status)
		else:
			text = "\r   {0}     {1}".format(" "*barLength,status)
	else:
		text += "[{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
	sys.stdout.write(text)
	sys.stdout.flush()

def ComputeVertEddy(v,t,p,p0=1e3,wave=0):
	""" Computes the vertical eddy components of the residual circulation,
		bar(v'Theta'/Theta_p). Either in real space, or a given wave number.
		Dimensions must be time x pres x lat x lon.
		Output dimensions are: time x pres x lat
		Output units are [v_bar] = [v], [t_bar] = [v*p]

		INPUTS:
			v    - meridional wind
			t    - temperature
			p    - pressure coordinate
			p0   - reference pressure for potential temperature
			wave - wave number (if >=0)
		OUPUTS:
			v_bar - zonal mean meridional wind [v]
			t_bar - zonal mean vertical eddy component <v'Theta'/Theta_p> [v*p]
	"""
	#
	# some constants
	from .constants import kappa
	#
	# pressure quantitites
	pp0 = (p0/p[np.newaxis,:,np.newaxis,np.newaxis])**kappa
	dp  = np.gradient(p)[np.newaxis,:,np.newaxis]
	# convert to potential temperature
	t = t*pp0 # t = theta
	# zonal means
	v_bar = np.nanmean(v,axis=-1)
	t_bar = np.nanmean(t,axis=-1) # t_bar = theta_bar
	# prepare pressure derivative
	dthdp = np.gradient(t_bar,edge_order=2)[1]/dp # dthdp = d(theta_bar)/dp
	dthdp[dthdp==0] = np.nan
	# time mean of d(theta_bar)/dp
	dthdp = np.nanmean(dthdp,axis=0)[np.newaxis,:]
	# now get wave component
	#if isinstance(wave,list):
	#	t = np.sum(GetWaves(v,t,wave=-1,do_anomaly=True)[:,:,:,wave],axis=-1)
	if wave == 0:
		v = GetAnomaly(v) # v = v'
		t = GetAnomaly(t) # t = t'
		t = np.nanmean(v*t,axis=-1) # t = bar(v'Th')
	else:
		t = GetWaves(v,t,wave=wave,do_anomaly=True) # t = bar(v'Th'_{k=wave})
		if wave < 0:
			dthdp = np.expand_dims(dthdp,-1)
	t_bar = t/dthdp # t_bar = bar(v'Th')/(dTh_bar/dp)
	#
	return v_bar,t_bar


##############################################################################################
def ComputeWstarXr(omega, temp, vcomp, pres='level', lon='lon', lat='lat', time='time', ref='mean', p0=1e3, is_Pa='omega'):
	"""Computes the residual upwelling w*. omega, temp, vcomp are xarray.DataArrays.

		Output units are the same as the units of omega, and the pressure coordinate is expected in hPa, latitude in degrees.

		INPUTS:
			omega - pressure velocity. xarray.DataArray
			temp  - temperature. xarray.DataArray
			vcomp - meridional velocity. xarray.DataArray
			pfull - name of pressure coordinate.
			lon   - name of longitude coordinate
			lat   - name of latitude coordinate
			time  - name of time coordinate
			ref   - how to treat dTheta/dp:
				- 'rolling-X' : centered rolling mean over X days
				- 'mean'      : full time mean
			p0    - pressure basis to compute potential temperature [hPa]
			is_Pa - correct for pressure units in variables:
				- None: omega, p0 and pres are all in hPa or all in Pa
				- 'omega': omega is in Pa/s but pres and p0 in hPa
				- 'pres' : omega is in hPa/s, p0 in hPa, but pres in Pa
		OUTPUTS:
			residual pressure velocity, same units as omega
	"""
	import numpy as np

	a0    = 6371000.

	# spherical geometry
	coslat = np.cos(np.deg2rad(omega[lat]))
	R = a0*coslat
	R = 1./R
	# correct for units: hPa<->Pa
	if is_Pa is not None:
		if is_Pa.lower() == 'omega':
			R = R*100
		elif is_Pa.lower() == 'pres':
			R  = R*0.01
			p0 = p0*100
        # correct for units: degrees<->radians
	R = R*180/np.pi
	# compute thickness weighted meridional heat flux
	_,vt_bar, dtdp_bar = ComputeVertEddyXr(vcomp, temp, pres, p0, lon, time, ref)
	# get the meridional derivative
	vt_bar = (coslat*vt_bar).differentiate(lat)
	# compute zonal mean upwelling
	w_bar = omega.mean(lon)
	# put it all together
	return w_bar + R*vt_bar

##############################################################################################
# @jit
def FlexiGradPhi(data,dphi):
	if len(data.shape) == 3:
		grad = np.gradient(data,edge_order=2)[2]
	else:
		grad = np.gradient(data,edge_order=2)[1]
	return grad/dphi
# @jit
def FlexiGradP(data,dp):
	if len(data.shape) == 3:
		grad = np.gradient(data,edge_order=2)[1]
	else:
		grad = np.gradient(data,edge_order=2)[0]
	return grad/dp
