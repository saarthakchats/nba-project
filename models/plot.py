#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Define years for each era
years_pre = list(range(1985, 2013))    # 1985 to 2012 (28 years)
years_post = list(range(2013, 2024))    # 2013 to 2023 (11 years)

# Arbitrary FG₃M coefficient values for 1985-2012 within [0.001, 0.09]
coeffs_pre = [
    0.005,  # 1985
    0.009,  # 1986
    0.0058, # 1987
    0.01,  # 1988
    0.0072, # 1989
    0.007,  # 1990
    0.01,  # 1991
    0.006, # 1992
    0.012, # 1993
    0.01,  # 1994
    0.005, # 1995
    0.0095, # 1996
    0.0098, # 1997
    0.0100, # 1998
    0.0102, # 1999
    0.0110, # 2000
    0.0132, # 2001
    0.0115, # 2002
    0.0140, # 2003
    0.0132, # 2004
    0.0230, # 2005
    0.0240, # 2006
    0.0190, # 2007
    0.0260, # 2008
    0.0470, # 2009
    0.0480, # 2010
    0.0500, # 2011
    0.0690  # 2012 (a jump to the upper limit)
]

# Arbitrary FG₃M coefficient values for 2013-2023 within [0.09, 0.2]
coeffs_post = [
    0.080,  # 2013
    0.10,  # 2014
    0.112,  # 2015 (small dip)
    0.130,  # 2016
    0.127,  # 2017
    0.142,  # 2018
    0.135,  # 2019
    0.150,  # 2020
    0.165,  # 2021
    0.180,  # 2022
    0.200   # 2023
]

# Combine the years and coefficients for the full period
years_all = years_pre + years_post
coeffs_all = coeffs_pre + coeffs_post

# Plot the evolution of the FG3M coefficient
plt.figure(figsize=(10, 6))
plt.plot(years_all, coeffs_all, marker='o', linestyle='-', color='blue', label='FG₃M Coefficient')
plt.axvline(x=2013, color='red', linestyle='--', label='2013 Threshold')
plt.title('Evolution of FG₃M Coefficient (1985-2023)')
plt.xlabel('Year')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
