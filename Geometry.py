import numpy as np
import sympy as sp
import pandas as pd 
import matplotlib.pyplot as plt

#-----------------  SURFACE PARAMETRIZATION-----------

def sphere_sp(x,y):
    '''
        Define the shape of the surface of the sphere using only simpy function as z(x,y)
    '''
    return  sp.sqrt(r**2 - x**2 - y**2)



def cylinder_sp(x,y):
    '''
        Define the shape of the surface using only simpy function as z(x,y)
    '''
    return  sp.sqrt(r-x**2)

def parametrization_sphere_sp(theta,phi):
    '''
        Define the shape of the sphere surface using only simpy function in spheric coordinates
    '''
    return sp.Matrix([r*sp.cos(theta)*sp.cos(phi), r*sp.cos(theta)*sp.sin(phi), r*sp.sin(theta)])


def parametrization_cyl_sp(theta,z):
    '''
        Define the shape of the cylinder surface using only simpy function in cylindric coordinates
    '''
    return sp.Matrix([r*sp.cos(theta), r*sp.sin(theta), z])

def cylinder_pert_sp(theta,z):
    '''
        Define the shape of the perturbated cylinder surface using only simpy function in cylindric coordinates
    '''
    return sp.Matrix([(r + b * sp.cos(k*z)) * sp.cos(theta), (r + b * sp.cos(k*z)) * sp.sin(theta)  ,  z])



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



def pressure_comp_xy(func_z,data):
    '''
        Compute the pressure (mean curvature) apart from a constant given a surface z(x,y). 
        
        Arguments:
            func_z: is the parametrization of the surface as (x,y,z(x,y)) using sympy functions
            data: an array with data with shape (len(data),3)
        Returns an array with the local pressure
    '''
    pressure=[]
    x, y = sp.symbols('x y')
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
    x, y = sp.symbols('x y')
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

def pressure_theory_comp(a,func_z):
    '''
        Compute the mean curvature given a surface parametrization as a math formula. 
        
        Arguments:
            x, y: sympy simbol for the two main variables of the system
            func_z: is the parametrization of the surface as (f(x,y),g(x,y),z(x,y)) using sympy functions
            
        Returns a sympy formula for the mean curvature
    '''

    x = a[0]
    ym= a[1]
    
    f = func_z(a)

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    df_dxx = sp.diff(f, x,x)
    df_dyy = sp.diff(f, y,y)
    df_dxy = sp.diff(f, x,y)
    

    #computation of normal direction
    normal = df_dx.cross(df_dy) 
    normal = normal / normal.norm()

    #computation of coefficients
    E = df_dx.dot(df_dx)
    F = df_dx.dot(df_dy)
    G = df_dy.dot(df_dy)
    L = df_dxx.dot(normal)
    M = df_dxy.dot(normal)
    N = df_dyy.dot(normal)

    pressure_val = (E*N + G*L - 2*F*M)/(E*G-F**2)
    
    return pressure_val
