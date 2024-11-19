# pylint: skip-file

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def PlotEPfluxArrows(x,y,ep1,ep2,fig,ax,xlim=None,ylim=None,xscale='linear',yscale='linear',invert_y=True, newax=False, pivot='tail',scale=None,quiv_args=None):
	"""Correctly scales the Eliassen-Palm flux vectors for plotting on a latitude-pressure or latitude-height axis.
		x,y,ep1,ep2 assumed to be xarray.DataArrays.

	INPUTS:
		x	: horizontal coordinate, assumed in degrees (latitude) [degrees]
		y	: vertical coordinate, any units, but usually this is pressure or height
		ep1	: horizontal Eliassen-Palm flux component, in [m2/s2]. Typically, this is ep1_cart from
				   ComputeEPfluxDiv()
		ep2	: vertical Eliassen-Palm flux component, in [U.m/s2], where U is the unit of y.
				   Typically, this is ep2_cart from ComputeEPfluxDiv(), in [hPa.m/s2] and y is pressure [hPa].
		fig	: a matplotlib figure object. This figure contains the axes ax.
		ax	: a matplotlib axes object. This is where the arrows will be plotted onto.
		xlim	: axes limits in x-direction. If None, use [min(x),max(x)]. [None]
		ylim	: axes limits in y-direction. If None, use [min(y),max(y)]. [None]
		xscale	: x-axis scaling. currently only 'linear' is supported. ['linear']
		yscale	: y-axis scaling. 'linear' or 'log' ['linear']
		invert_y: invert y-axis (for pressure coordinates). [True]
		newax	: plot on second y-axis. [False]
		pivot	: keyword argument for quiver() ['tail']
		scale	: keyword argument for quiver(). Smaller is longer [None]
				  besides fixing the length, it is also usefull when calling this function inside a
				   script without display as the only way to have a quiverkey on the plot.
               quiv_args: further arguments passed to quiver plot.

	OUTPUTS:
	   Fphi*dx : x-component of properly scaled arrows. Units of [m3.inches]
	   Fp*dy   : y-component of properly scaled arrows. Units of [m3.inches]
	   ax	: secondary y-axis if newax == True
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	#
	def Deltas(z,zlim):
		# if zlim is None:
		return np.max(z)-np.min(z)
		# else:
			# return zlim[1]-zlim[0]
	# Scale EP vector components as in Edmon, Hoskins & McIntyre JAS 1980:
	cosphi = np.cos(np.deg2rad(x))
	a0 = 6376000.0 # Earth radius [m]
	grav = 9.81
	# first scaling: Edmon et al (1980), Eqs. 3.1 & 3.13
	Fphi = 2*np.pi/grav*cosphi**2*a0**2*ep1 # [m3.rad]
	Fp   = 2*np.pi/grav*cosphi**2*a0**3*ep2 # [m3.hPa]
	#
	# Now comes what Edmon et al call "distances occupied by 1 radian of
	#  latitude and 1 [hecto]pascal of pressure on the diagram."
	# These distances depend on figure aspect ratio and axis scale
	#
	# first, get the axis width and height for
	#  correct aspect ratio
	width,height = GetAxSize(fig,ax)
	# we use min(),max(), but note that if the actual axis limits
	#  are different, this will be slightly wrong.
	delta_x = Deltas(x,xlim)
	delta_y = Deltas(y,ylim)
	#
	#scale the x-axis:
	if xscale == 'linear':
		dx = width/delta_x/np.pi*180
	else:
		raise ValueError('ONLY LINEAR X-AXIS IS SUPPORTED AT THE MOMENT')
	#scale the y-axis:
	if invert_y:
		y_sign = -1
	else:
		y_sign = 1
	if yscale == 'linear':
		dy = y_sign*height/delta_y
	elif yscale == 'log':
		dy = y_sign*height/y/np.log(np.max(y)/np.min(y))
	#
	# plot the arrows onto axis
	quivArgs = {'angles':'uv','scale_units':'inches','pivot':pivot}
	if quiv_args is not None:
		for key in quiv_args.keys():
			quivArgs[key] = quiv_args[key]
	if scale is not None:
		quivArgs['scale'] = scale
	if newax:
		ax = ax.twinx()
		ax.set_ylabel('pressure [hPa]')
	try:
		Q = ax.quiver(x,y,Fphi*dx,Fp*dy,**quivArgs)
	except:
		Q = ax.quiver(x,y,dx*Fphi.transpose(),dy*Fp.transpose(),**quivArgs)
	if scale is None:
		fig.canvas.draw() # need to update the plot to get the Q.scale
		U = Q.scale
	else:
		U = scale
	if U is not None: # when running inside a script, the figure might not exist and therefore U is None
		ax.quiverkey(Q,0.85,1.05,U/width,label=r'{0:.1e}$\,m^3$'.format(U),labelpos='E',coordinates='axes')
	if invert_y:
		ax.invert_yaxis()
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	ax.set_yscale(yscale)
	ax.set_xscale(xscale)
	#
	if newax:
		return Fphi*dx,Fp*dy,ax
	else:
		return Fphi*dx,Fp*dy

def GetAxSize(fig,ax,dpi=False):
	"""get width and height of a given axis.
	   output is in inches if dpi=False, in dpi if dpi=True
	"""
	bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
	width, height = bbox.width, bbox.height
	if dpi:
		width *= fig.dpi
		height *= fig.dpi
	return width, height



#-------------------------------------------------------------------------------------------------------------------


def ComputeEPfluxDivXr(u,v,t,lon='infer',lat='infer',pres='infer',time='time',ref='mean',w=None,do_ubar=False,wave=0):
	""" Compute the EP-flux vectors and divergence terms.

		The vectors are normalized to be plotted in cartesian (linear)
		coordinates, i.e. do not incluxde the geometric factor a*cos\\phi.
		Thus, ep1 is in [m2/s2], and ep2 in [hPa*m/s2].
		The divergence is in units of m/s/day, and therefore represents
		the deceleration of the zonal wind. This is actually the quantity
		1/(acos\\phi)*div(F).

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
	  div1 - horizontal EP-flux divergence, divided by acos\\phi [m/s/d]
	  div2 - horizontal EP-flux divergence , divided by acos\\phi [m/s/d]
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
	vbar,vertEddy = ComputeVertEddyXr(v,t,pres,p0,lon,time,ref,wave) # vertEddy = bar(v'Th'/(dTh_bar/dp))
	#
	## get zonal anomalies
	if isinstance(wave,list):
		upvp = GetWavesXr(u,v,dim=lon,wave=-1).sel(k=wave).sum('k')
	elif wave == 0:
		u = u - u.mean(lon)
		v = v - v.mean(lon)
		upvp = (u*v).mean(lon)
	else:
		upvp = GetWavesXr(u,v,dim=lon,wave=wave)
	#
	## compute the horizontal component
	if do_ubar:
		shear = ubar.differentiate(pres,edge_order=2) # [m/s.hPa]
	else:
		shear = 0.
	ep1_cart = -upvp + shear*vertEddy # [m2/s2 + m/s.hPa*m.hPa/s] = [m2/s2]
	#
	## compute vertical component of EP flux.
	## at first, keep it in Cartesian coordinates, ie ep2_cart = f [v'theta'] / [theta]_p + ...
	#
	ep2_cart = fhat*vertEddy # [1/s*m.hPa/s] = [m.hPa/s2]
	if w is not None:
		if isinstance(wave,list):
			w = GetWavesXr(u,w,dim=lon,wave=-1).sel(k=wave).sum('k')
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
	ep1_cart.name, ep1_cart.units = 'epfy', 'm2 s-2'
	ep2_cart.name. ep2_cart.units = 'epfz', 'm.hPa s-2'
	div1.name, div1.units = 'divFy', 'm s-2'
	div2.name, div2.units = 'divFz', 'm s-2'
	return ep1_cart.transpose(*new_order),ep2_cart.transpose(*new_order),div1.transpose(*new_order),div2.transpose(*new_order)


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
		vpTp = GetWavesXr(v,t,dim=lon,wave=-1).sel(k=wave).sum('k')
	elif wave == 0:
		vpTp = (v - v_bar)*(t - t_bar)
		vpTp = vpTp.mean(lon)  # vpTp = bar(v'Th')
	else:
		vpTp = GetWavesXr(v,t,dim=lon,wave=wave) # vpTp = bar(v'Th'_{k=wave})
	t_bar = vpTp/dthdp # t_bar = bar(v'Th')/(dTh_bar/dp)
	#
	return v_bar,t_bar


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
	for plev in ['level','pres','pfull','lev','plev', 'pressure']:
		if plev in ldims:
			indx = ldims.index(plev)
			dim_names['pres'] = odims[indx]
	return dim_names


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

