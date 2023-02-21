#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.linalg as npl
import time


def ForwardBackward_step(x,s,proxh,Df):
    """Computes a Forward-Backward step for a composite function F=f+h where
    f is differentiable and the proximal operator of h can be computed.
    
    Parameters
    ----------
        
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        proxh : operator
            Proximal operator of the non differentiable function h.
        Df : operator
            Derivative of the differentiable function f.
    
    Returns
    -------
        
        xp: array_like
            Final vector.
    """
    return proxh(x-s*Df(x),s)



def ForwardBackward(x,s,Niter,epsilon,Df,proxh,F=None,exit_crit=None,
                    extra_function=None,track_ctime=False):
    """Forward-Backward method applied to a composite function F=f+h where f
    is differentiable and the proximal operator of h can be computed.
    
    Parameters
    ----------
        
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator, optional
            Function to minimize. If the user gives F, the function will
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and x_{k-1}. The default is the
            norm of (x_k - x_{k-1}).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.

    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given."""
    if F is not None:cost = [F(x)]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,xm:npl.norm(x-xm,2)
    i = 0
    out = False
    while i < Niter and out == False:
        xm = np.copy(x)
        x = ForwardBackward_step(x,s,proxh,Df)
        out = exit_crit(x,xm) < epsilon
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
        i += 1
    output = (x,)
    if F is not None:output+=(np.array(cost),)
    if track_ctime:output+=(np.array(ctime),)
    if extra_function is not None:output+=(np.array(extra_cost))
    if len(output) == 1:output = x
    return output

def FISTA(x,s,Niter,epsilon,Df,proxh,alpha=3,F=None,exit_crit=None
          ,restarted=False, extra_function=None,track_ctime=False):
    """FISTA applied to a composite function F=f+h where f is differentiable
    and the proximal operator of h can be computed.
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        F : operator, optional
            Function to minimize. If the user gives F, the function will 
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm 
            of (x_k - y_k).
        restarted : boolean, optional
            Parameter which specifies if FISTA will be restarted. If it is
            True the function FISTA returns an additional boolean "out" which
            reports if the exit condition was satisfied . The purpose is to
            avoid to check the exit condition again after running FISTA. The
            default value is False.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given.
        out : boolean, optional
            Parameter reporting if the exit condition is satisfied at the last
            iterate. It is returned if restarted is True which is not the case
            by default."""
    if F is not None and not restarted:cost = [F(x)]
    if F is not None and restarted:cost = []
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    y = np.copy(x)
    i = 0
    out = False
    while i < Niter and out == False:
        i += 1
        xm = np.copy(x)
        x = ForwardBackward_step(y,s,proxh,Df)
        out = exit_crit(x,y)<epsilon
        y = x+(i-1)/(i+alpha-1)*(x-xm)
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if restarted:output += (out,)
    if len(output) == 1:output = x
    return output


def VFISTA(x,s,Niter,epsilon,Df,proxh,alpha,F=None,exit_crit=None
          ,restarted=False, extra_function=None,track_ctime=False):
    """V-FISTA applied to a composite function F=f+h where f is differentiable
    and the proximal operator of h can be computed (see "An introduction to 
    continuous optimization for imaging." by Amir Beck).
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        alpha : float
            Inertial parameter involved in the step 
            y_k=x_k+alpha*(x_k-x_{k-1}).
        F : operator, optional
            Function to minimize. If the user gives F, the function will 
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm 
            of (x_k - y_k).
        restarted : boolean, optional
            Parameter which specifies if FISTA will be restarted. If it is
            True the function FISTA returns an additional boolean "out" which
            reports if the exit condition was satisfied . The purpose is to
            avoid to check the exit condition again after running FISTA. The
            default value is False.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given.
        out : boolean, optional
            Parameter reporting if the exit condition is satisfied at the last
            iterate. It is returned if restarted is True which is not the case
            by default."""
    if F is not None and not restarted:cost = [F(x)]
    if F is not None and restarted:cost = []
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        extratime = 0
        ctime = [0]
        t0 = time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    y = np.copy(x)
    i = 0
    out = False
    while i < Niter and out == False:
        i += 1
        xm = np.copy(x)
        x = ForwardBackward_step(y,s,proxh,Df)
        out = exit_crit(x,y)<epsilon
        y = x+alpha*(x-xm)
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extratime += time.perf_counter()-t_temp
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if restarted:output += (out,)
    if len(output) == 1:output = x
    return output

    
def FISTA_fixed_restart(x,s,n_r,Niter,epsilon,Df,proxh,alpha=3,F=None,
                        exit_crit=None,extra_function=None,track_ctime=False):
    """Restarted version of FISTA (a restart occurs every n_r iterations).
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        n_r : integer
            Number of iterations between each restart.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        alpha : float, optional
            Inertial parameter involved in the step
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        F : operator, optional
            Function to minimize. If the user gives F, the function will
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm of
            (x_k - y_k).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given."""
    if F is not None:cost = [F(x)]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    if track_ctime:ctime = [0]
    i = 0
    out = False
    while i < Niter and out == False:
        outputs = FISTA(x,s,np.minimum(Niter-i,n_r),epsilon,Df,proxh,alpha,F,
                        exit_crit,restarted=True,extra_function=extra_function
                        ,track_ctime=track_ctime)
        x = outputs[0]
        j = 1
        if F is not None:
            cost_temp = outputs[j]
            cost = np.concatenate((cost,cost_temp))
            j+=1
        if track_ctime:
            ctime_temp = outputs[j]
            ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
            j+=1
        if extra_function is not None:
            extra_cost_temp = outputs[j]
            extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
            j+=1
        out = outputs[j]        
        i += np.minimum(Niter-i,n_r)
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if len(output) == 1:output = x
    return output
        
    
def FISTA_automatic_restart(x,s,Niter,epsilon,Df,proxh,F,alpha=3,C=6.38,
                            exit_crit=None,out_cost=True,estimated_ratio=1,
                            out_mu=False, extra_function=None,
                            track_ctime=False):
    """Automatic restart of FISTA (method introduced in "FISTA restart using
    an automatic estimation of the growth parameter").
    
    Parameters
    ----------
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        F : operator
            Function to minimize.
        alpha : float, optional
            Inertial parameter involved in the step 
            y_k=x_k+k/(k+alpha)*(x_k-x_{k-1}). The default value is 3.
        C : float, optional
            Parameter of the restart method. Large values of C ensure frequent
            restarts. The default value is 6.38 (theoretically optimal).
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm of
            (x_k - y_k).
        out_cost : boolean, optional
            Parameter which states if the function returns the evolution of 
            F(x_k). The default value is True.
        estimated_ratio : float, optional
            Low estimation of the condition number. The default value is 1.
        out_mu : boolean, optional
            Parameter which states if the function returns the successive 
            estimations of the growth parameter mu. The default value is False.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            out_cost is True.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given.
        growth_estimates : array_like, optional
            Array containing the estimations of the growth parameter mu. It is
            returned if out_mu is True."""
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    objective = None
    if out_cost:objective = F
    if out_mu:growth_estimates = np.array([])
    if extra_function is not None: extra_cost = np.array([extra_function(x)])
    if track_ctime:t0 = time.perf_counter()
    i = 0
    n = int(2*C*np.sqrt(estimated_ratio))
    F_tab = [F(x)]
    n_tab = [n]
    if track_ctime:ctime = [time.perf_counter()-t0]
    if out_cost: cost = [F_tab[0]]
    outputs = FISTA(x,s,n,epsilon,Df,proxh,alpha,objective,exit_crit,
                    restarted=True,extra_function=extra_function,
                    track_ctime=track_ctime)
    x = outputs[0]
    j = 1
    if out_cost:
        cost_temp = outputs[j]
        cost = np.concatenate((cost,cost_temp))
        j+=1
    if track_ctime:
        ctime_temp = outputs[j]
        ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
        j+=1
    if extra_function is not None:
        extra_cost_temp = outputs[j]
        extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
        j+=1
    out = outputs[j]
    if track_ctime:t_temp = time.perf_counter()
    i += n
    F_tab = np.concatenate((F_tab,[F(x)]))
    n_tab += [n]
    while (i < Niter) and out==False:
        if track_ctime:ctime[-1] += time.perf_counter()-t_temp
        outputs = FISTA(x,s,np.minimum(n,Niter-i),epsilon,Df,proxh,alpha,
                        objective,exit_crit,restarted=True,
                        extra_function=extra_function,track_ctime=track_ctime)
        x = outputs[0]
        j = 1
        if out_cost:
            cost_temp = outputs[j]
            cost = np.concatenate((cost,cost_temp))
            j+=1
        if track_ctime:
            ctime_temp = outputs[j]
            ctime = np.concatenate((ctime,ctime_temp[1:]+ctime[-1]))
            j+=1
        if extra_function is not None:
            extra_cost_temp = outputs[j]
            extra_cost = np.concatenate((extra_cost,extra_cost_temp[1:]))
            j+=1
        out = outputs[j]
        
        if track_ctime:t_temp = time.perf_counter()
        i += np.minimum(n,Niter-i)
        F_tab = np.concatenate((F_tab,[F(x)]))
        tab_mu = (4/(s*(np.array(n_tab)[:-1]+1)**2)*(F_tab[:-2]-F_tab[-1])/
                  (F_tab[1:-1]-F_tab[-1]))
        mu = np.min(tab_mu)
        if out_mu:growth_estimates = np.r_[growth_estimates,mu]
        if (n <= C/np.sqrt(s*mu)):
            n = 2*n
        n_tab += [n]
    output = (x,)
    if out_cost:output += (cost,)
    if track_ctime:output += (ctime,)
    if extra_function is not None:output += (extra_cost,)
    if out_mu:output += (growth_estimates,)
    if len(output) == 1:output = x
    return output
    


def FISTA_Hessian(x,s,Niter,epsilon,Df,proxh,alpha=3,beta=None,F=None,
                  exit_crit=None,extra_function=None,track_ctime=False):
    """Adapted version of Inertial Gradient Algorithm with Hessian Damping
    introduced in "First-order optimization algorithms via inertial systems
    with Hessian driven damping".
    It is applied to a composite function F=f+h where f is differentiable and
    the proximal operator of h can be computed.
    
    Parameters
    ----------
    
        x : array_like
            Initial vector.
        s : float
            Step size of the method.
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        alpha : float, optional
            Inertial parameter. The default value is 3.
        beta : float, optional
            Parameter related to the Hessian damping. The default value is
            2sqrt(s) (the maximum value ensuring theoretical convergence).
        F : operator, optional
            Function to minimize. If the user gives F, the function will
            compute the value of F at each iterate.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k. The default is the norm of
            (x_k - y_k).
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
        
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            F is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given.
        """
    if F is not None:cost = [F(x)]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        ctime = [0]
        extra_time = 0
        t0=time.perf_counter()
    if exit_crit is None:exit_crit = lambda x,y:npl.norm(x-y,2)
    if beta is None: beta = 2*np.sqrt(s)
    y = np.copy(x)
    Dfx = Df(x)
    i = 0
    out = False
    while i < Niter and out == False:
        i += 1
        xm = np.copy(x)
        x = ForwardBackward_step(y,s,proxh,Df)
        Dfxm = Dfx
        Dfx = Df(x)
        y = (x+(i-1)/(i+alpha-1)*(x-xm)
             -beta*np.sqrt(s)*(Dfx-Dfxm)-beta*np.sqrt(s)/i*Dfxm)
        out = exit_crit(x,y)<epsilon
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp-extra_time-t0]
        if F is not None:cost += [F(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if track_ctime:extra_time += time.perf_counter()-t_temp
    output = (x,)
    if F is not None:output += (np.array(cost),)
    if track_ctime:output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost),)
    if len(output) == 1:output = x
    return output

def Breg(x,y,f,Df,sp=None,out_f=False):
    """Computation of the Bregman distance of f between x and y.
        
    Parameters
    ----------
    
        x,y : array_like
            Entry vectors.
        f : operator
            Entry function.
        Df : operator
            Gradient of f.
        sp : operator, optional
            Scalar product of reference.
        out_f : boolean, optional
            Parameter ensuring that the function returns f(x) if it is True.
            The default value is False.

    Returns
    -------
        
        D : float
            Bregman distance of f between x and y.
        fx : float, optional
            Value of f(x). It is returned if out_f is True."""
    if sp is None:sp = lambda x,y:np.dot(x,y)
    if out_f:
        fx = f(x)
        return fx-f(y)-sp(Df(y),x-y),fx
    else:return f(x)-f(y)-sp(Df(y),x-y)

def FISTA_BT(x,L0,rho,delta,Niter,epsilon,f,Df,proxh,h=None,exit_crit=None,
             sp=None,out_L=False,out_ite=False,restarted=False,
             exit_norm=False,extra_function=None,track_ctime=False):
    """FISTA with backtracking (version introduced in "Backtracking strategies
    for accelerated descent methods with smooth composite objectives" in the
    non-strongly convex case) applied to a composite function F=f+h where f is
    differentiable and the proximal operator of h can be computed.
    
    Parameters
    ----------
        x : array_like
            Initial vector.
        L0 : float
            Estimation of the Lipschitz constant L of the gradient of f.
        rho : float
            Armijo parameter, should be in (0,1). We recommand to choose rho
            in [0.8,0.9].
        delta : float
            Increasing parameter, should be in (0,1]. We recommand to chose
            delta in [0.95,1].
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        f : operator
            Differentiable part of the function F to minimize.
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        h : operator, optional
            Non differentiable part of the function F to minimize. If it is
            given by the user, the function computes F at each iterates.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k the current step size.
            The default is the norm of (x_k - y_k).
        sp : operator, optional
            Scalar product of reference.
        out_L : boolean, optional
            If it is True, the function returns the array of the successive
            estimations of L. The default value is False.
        out_ite : boolean, optional
            If it is True, the function returns the array of backtracking
            iterations per FISTA iteration. The default value is False.
        restarted : boolean, optional
            If it is True, the function returns the boolean "out" which
            specifies if the exit condition was satisfied, the value of f at
            the last iterate and the successive estimation of L. The default
            value is False.
        exit_norm : boolean, optional
            Parameter which states if the exit criteria is the euclidean norm
            of (x_k-y_k)/tau associated to the scalar product sp. The purpose
            is to save computations. The default value is False.
        extra_function : operator, optional
            Additional function to compute at each iterate.
        track_ctime : boolean, optional
            Parameter which states if the function returns an array containing
            the computation time at each iteration. The default value is False.
            
    Returns
    -------
        
        x : array_like
            Last iterate given by the method.
        cost : array_like, optional
            Array containing the value of F at each iterate. It is returned if
            h is given by the user.
        ctime : array_like, optional
            Array containing the computation time required at each iterate. 
            The last coefficient is the total computation time. It is returned
            if track_ctime is True.
        extra_cost : array_like, optional
            Array containing the value of extra_function at each iterate if a 
            function is given.
        tab_L : array_like, optional
            Value of the estimates of L at each iterates. It is returned if
            restarted is True or out_L is True.
        tab_j : array_like, optional
            Array containing the number of backtracking iterations per global
            iteration. It is returned if out_ite is True.
        fx : float, optional
            Value of f at the last iterate. It is returned if restarted is
            True.
        out : boolean, optional
            Parameter reporting if the exit condition is satisfied at the last
            iterate. It is returned if restarted is True.
        """
    if h is not None and not restarted:
        cost=[f(x)+h(x)]
    if h is not None and restarted:
        cost=[]
    if sp is None:sp=lambda x,y:np.dot(x,y)
    if exit_crit is None:
        exit_crit=lambda x,y:np.sqrt(sp(x-y,x-y))
        exit_norm=True
    if out_L:tab_L=[L0]
    if restarted:tab_L=[]
    if out_ite and not restarted:tab_j=[0]
    if out_ite and restarted:tab_j=[]
    if extra_function is not None:extra_cost = [extra_function(x)]
    if track_ctime:
        t0 = time.perf_counter()
        extratime = 0
        ctime = [0]
    s=1/L0
    t=1
    i=0
    xm=np.copy(x)
    out=False
    forward=True#Parameter which states if the following iteration will perform an increment of the stepsize (s=s/delta)
    while i<Niter and out==False:
        if forward:s=s/delta
        cond=False
        j=0
        while cond==False:
            temp_s=rho**j*s
            temp_t=(1+np.sqrt(1+4*t**2*s/temp_s))/2
            temp_beta=(t-1)/temp_t
            temp_y=x+temp_beta*(x-xm)
            temp_x=ForwardBackward_step(temp_y,temp_s,proxh,Df)
            temp_norm=sp(temp_x-temp_y,temp_x-temp_y)
            if h is not None or restarted: # Condition = Do we need to keep the value of f(xk)?
                temp_breg,fx=Breg(temp_x,temp_y,f,Df,sp,True)
                cond=temp_breg<=temp_norm/(2*temp_s)
            else:
                temp_breg=Breg(temp_x,temp_y,f,Df,sp)
                cond=temp_breg<=temp_norm/(2*temp_s)
            j += 1
        if out_ite:tab_j+=[j]
        forward=temp_breg<=rho*temp_norm/(2*temp_s)
        s=temp_s
        t=temp_t
        xm=np.copy(x)
        x=np.copy(temp_x)
        if exit_norm:out=np.sqrt(temp_norm)<epsilon
        else:out=exit_crit(x,temp_y)<epsilon
        if track_ctime:
            t_temp = time.perf_counter()
            ctime += [t_temp - extratime - t0]
        if h is not None:cost+=[fx+h(x)]
        if extra_function is not None:extra_cost += [extra_function(x)]
        if out_L or restarted:tab_L+=[1/s]
        if track_ctime:extratime += time.perf_counter()-t_temp
        i+=1
    output = (x,)
    if h is not None:output += (np.array(cost),)
    if track_ctime: output += (np.array(ctime),)
    if extra_function is not None:output += (np.array(extra_cost))
    if out_L or restarted:output += (np.array(tab_L),)
    if out_ite:output += (np.array(tab_j),)
    if restarted:output += (fx,out,)
    if len(output)==1:output=x
    return output

def Free_FISTA(x,L0,rho,delta,Niter,epsilon,f,h,Df,proxh,C=None,exit_crit=None,
               sp=None,out_L=False,out_condition=False,out_ite=False,
               estimated_ratio=1,exit_norm=False):
    """Automatic FISTA restart with backtracking applied to a composite
    function F=f+h where f is differentiable and the proximal operator of h
    can be computed.
    
    Parameters
    ----------
        x : array_like
            Initial vector.
        L0 : float
            Estimation of the Lipschitz constant L of the gradient of f.
        rho : float
            Armijo parameter, should be in (0,1). We recommand to choose rho
            in [0.8,0.9].
        delta : float
            Increasing parameter, should be in (0,1]. We recommand to chose
            delta in [0.95,1].
        Niter : integer
            Maximum number of iterations.
        epsilon: float
            Expected accuracy for the given exit criteria.
        f : operator
            Differentiable part of the function F to minimize.
        h : operator
            Non differentiable part of the function F to minimize. 
        Df : operator
            Derivative of the differentiable function f.
        proxh : operator
            Proximal operator of the non differentiable function h.
        C : float, optional
            Parameter of the restart method. Large values of C ensure frequent
            restarts. The default value is C=6.38/sqrt(rho) which is
            theoretically optimal.
        exit_crit : operator, optional
            Exit criteria, function of x_k and y_k.
            The default is the norm of (x_k - y_k).
        sp : operator, optional
            Scalar product of reference.
        out_L : boolean, optional
            If it is True, the function returns the array of the successive
            estimations of L. The default value is False.
        out_condition : boolean, optional
            Parameter which states if the function returns the successive
            estimations of the condition number. The default value is False.
        out_ite : boolean, optional
            If it is True, the function returns the array of backtracking
            iterations per global iteration. The default value is False.
        exit_norm : boolean, optional
            Parameter which states if the exit criteria is the euclidean norm
            of (x_k-y_k) associated to the scalar product sp. The purpose
            is to save computations. The default value is False.
        estimated_ratio : flat, optional
            Low estimation of the condition number. The default value is 1.
            
    Returns
    -------
        x : array_like
            Last iterate given by the method.
        cost : array_like
            Array containing the value of F at each iterate. It is returned if
            out_cost is True.
        tab_L : array_like, optional
            Value of the estimates of L at each iterates. It is returned if
            restarted is True or out_L is True.
        condition_estimates : array_like, optional
            Array containing the estimations of the condition number. It is
            returned if out_condition is True.
        tab_j : array_like, optional
            Array containing the number of backtracking iterations per global
            iteration. It is returned if out_ite is True.
       """
    if sp is None:sp=lambda x,y:np.dot(x,y)
    if exit_crit is None:
        exit_crit=lambda x,y:np.sqrt(sp(x-y,x-y))
        exit_norm=True
    if C is None:C=6.38/np.sqrt(rho)
    if out_L:tab_L=[L0]
    if out_ite:tab_ite=[0]
    if out_condition:condition_estimates=[]
    t0 = time.perf_counter()
    i=0
    n=int(2*C*np.sqrt(estimated_ratio))
    F_tab=[f(x)+h(x)]
    n_tab=[n]
    cout=[F_tab[0]]
    ctime = [time.perf_counter()-t0]
    temp=FISTA_BT(x,L0,rho,delta,n,epsilon,f,Df,proxh,h,exit_crit,sp,out_L,out_ite,restarted=True,exit_norm=exit_norm)
    t_temp = time.perf_counter()
    x,cout_temp,ctime_temp,out,fx,tab_L_temp=temp[:6]
    cout=np.concatenate((cout,cout_temp))
    ctime = np.concatenate((ctime,np.array(ctime_temp[1:])+ctime[-1]))
        
    
    if out_L:tab_L=np.concatenate((tab_L,tab_L_temp))
    if out_ite:tab_ite=np.concatenate((tab_ite,temp[-1]))
    L_tilde=tab_L_temp[-1]
    i+=n
    F_tab=np.concatenate((F_tab,[fx+h(x)]))
    n_tab+=[n]
    while (i<Niter) and out==False:
        ctime[-1]+=time.perf_counter()-t_temp
        temp=FISTA_BT(x,L_tilde,rho,delta,np.minimum(n,Niter-i),epsilon,f,Df,proxh,h,exit_crit,sp,out_L,out_ite,restarted=True,exit_norm=exit_norm)
        t_temp = time.perf_counter()
        x,cout_temp,ctime_temp,out,fx,tab_L_temp=temp[:6]
        cout=np.concatenate((cout,cout_temp))
        ctime = np.concatenate((ctime,np.array(ctime_temp[1:])+ctime[-1]))
        
        if out_L:tab_L=np.concatenate((tab_L,tab_L_temp))
        if out_ite:tab_ite=np.concatenate((tab_ite,temp[-1]))
        L_tilde=tab_L_temp[-1]#(len(tab_L)+1)**2/(np.sum(1/np.sqrt(tab_L)))**2
        i+=np.minimum(n,Niter-i)
        F_tab=np.concatenate((F_tab,[fx+h(x)]))
        tab_q=4/(rho*(np.array(n_tab)[:-1]+1)**2)*(F_tab[:-2]-F_tab[-1])/(F_tab[1:-1]-F_tab[-1])
        q=np.min(tab_q)
        if (n<=C/np.sqrt(q)):
            n=2*n
        n_tab+=[n]
        if out_condition:condition_estimates=np.concatenate((condition_estimates,[q]))
    output=(x,cout,ctime)
    if out_L:output=output+(np.array(tab_L),)
    if out_condition:output=output+(np.array(condition_estimates),)
    if out_ite:output=output+(np.array(tab_ite),)
    if len(output)==1:output=x
    return output
