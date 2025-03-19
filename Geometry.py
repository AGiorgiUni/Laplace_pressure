import numpy as np
import sympy as sp
import pandas as pd 
import matplotlib.pyplot as plt

def surface_derivatives(param):
    '''
        Compute the derivatives of the surface z(x,y). 
        
        Arguments:
            param is the parametrization of the surface as (f(x,y),g(x,y),z(x,y)) using sympy functions
        Returns all the derivatives 
    '''
    x, y = sp.symbols('x y')
    f = param(x,y)

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    df_dxx = sp.diff(f, x,x)
    df_dyy = sp.diff(f, y,y)
    df_dxy = sp.diff(f, x,y)
    return df_dx,df_dy,df_dxx,df_dyy,df_dxy



def sphere_sp(x,y):
    '''
        Define the shape of the surphace using only simpy function
    '''
    return  sp.sqrt(r**2 - x**2 - y**2)



def cylinder_sp(x,y):
    '''
        Define the shape of the surphace using only simpy function
    '''
    return  sp.sqrt(r-x**2)



def pressure_comp_xy(func_z,data):
    '''
        Compute the pressure (mean curvature) apart from a constant given a surface z(x,y). 
        
        Arguments:
            func_z: is the parametrization of the surface as (x,y,z(x,y)) using sympy functions
            data: an array with data with shape (len(data),3)
        Returns an array with the local pressure
    '''
    pressure=[]
    df_dx,df_dy,df_dxx,df_dyy,df_dxy = surface_derivatives(func_z)
    for i in range(len(data)):

        f_x = df_dx.subs({x: data[i,0], y: data[i,1] })
        f_y = df_dy.subs({x: data[i,0], y: data[i,1] })
        f_xx = df_dxx.subs({x: data[i,0], y: data[i,1] })
        f_yy = df_dyy.subs({x: data[i,0], y: data[i,1] })
        f_xy = df_dxy.subs({x: data[i,0], y: data[i,1] })

        pressure_val = (f_yy*(1+f_x**2)+ f_xx * (1+f_y**2) - 2*f_x*f_y*f_xy)/(np.pow(1+f_x**2+f_y**2,3/2))
        pressure.append((pressure_val))
    return pressure


def pressure_comp(func_z,data):
    '''
        Compute the mean curvature given a surface parametrization. 
        
        Arguments:
            func_z: is the parametrization of the surface as (f(x,y),g(x,y),z(x,y)) using sympy functions
            data: an array with data with shape (len(data),3)
        Returns an array with the local pressure
    '''
    pressure=[]
    df_dx,df_dy,df_dxx,df_dyy,df_dxy = surface_derivatives(func_z)
    for i in range(len(data)):

        #computation of the local derivatives
        f_x = df_dx.subs({x: data[i,0], y: data[i,1] })
        f_y = df_dy.subs({x: data[i,0], y: data[i,1] })
        f_xx = df_dxx.subs({x: data[i,0], y: data[i,1] })
        f_yy = df_dyy.subs({x: data[i,0], y: data[i,1] })
        f_xy = df_dxy.subs({x: data[i,0], y: data[i,1] })

        #computation of normal direction
        normal = f_x.cross(f_y) 
        normal = normal / normal.norm()

        #computation of coefficients
        E = f_x.dot(f_x)
        F = f_x.dot(f_y)
        G = f_y.dot(f_y)
        L = f_xx.dot(normal)
        M = f_xy.dot(normal)
        N = f_yy.dot(normal)

        pressure_val = (E*N + G*L - 2*F*M)/(E*G-F**2)
        pressure.append(pressure_val)
    return pressure