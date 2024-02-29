"""

Test script for HSSM for HDDM tutorial at Radboud University, February 2024

contact: seandamiandevine@gmail.com

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from IPython.display import Image ## for model graph with graphviz

import hssm 

hssm.set_floatX("float32")

## Load data from Cavanah et al.
dat = hssm.load_data('cavanagh_theta')

# ****************************************************************************
# *                          Fit basic DDM model                             *
# ****************************************************************************

simple_ddm_model = hssm.HSSM(
	data=dat, 
	model='ddm'
	)

simple_ddm_model.sample(draws=1000, 
	tune=1000, 
	chains=3, 
	cores=3
	)

simple_ddm_model.summary()


simple_ddm_model.plot_posterior_predictive()
plt.show()

simple_ddm_model.plot_trace()
plt.show()



# ****************************************************************************
# *               HDDM regression (non-hiearchical)                          *
# ****************************************************************************

dat['conf01'] = (dat['conf']=='HC').astype(float)

param_v = {
    "name": "v",
    "formula": "v ~ 1 + conf01",
    "prior": {
        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.25},
        "conf01": {"name": "Normal", "mu": 0.0, "sigma": 0.25},
    },
}



hddm_regression_nh = hssm.HSSM(
	data=dat, 
	model='ddm', 
	include=[param_v]
	)

ddm_graph = hddm_regression_nh.graph()


ddm_graph.view()



hddm_regression_nh.sample(draws=1000, 
	tune=1000, 
	chains=3,
	 cores=3
	 )

hddm_regression_nh.summary()



hddm_regression_nh.plot_trace();
plt.show()



# ****************************************************************************
# *               HDDM regression (hiearchical)                              *
# ****************************************************************************

param_v = {
    "name": "v",
    "formula": "v ~ 1 + theta + (1+theta|participant_id)",
    "link": "identity",
    "prior": {
        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.25},
        "theta": {"name": "Normal", "mu": 0.0, "sigma": 0.25},
        ## random effects must include hyperpriors
        "1|participant_id":
        	{"name": "Normal", "mu": 0.0, 
            "sigma": {"name": "HalfNormal", "sigma": 0.2} # this is a hyperprior
            },
        "theta|participant_id":
        	{"name": "Normal", "mu": 0.0, 
            "sigma": {"name": "HalfNormal", "sigma": 0.2} # this is a hyperprior
            },
    	},
    }


hddm_regression_h = hssm.HSSM(
	data=dat, 
	model='ddm', 
	include=[param_v]
	)


hddm_regression_h.sample(draws=1000, tune=1000, chains=3, cores=3)


hddm_regression_h.summary()


# ****************************************************************************
# *                              Angle DDM                                   *
# ****************************************************************************


angle_ddm_model = hssm.HSSM(
	data=dat, 
	model='angle'
	)

angle_ddm_model.sample(draws=1000, 
	tune=1000, 
	chains=3, 
	cores=3
	)

angle_ddm_model.summary()
