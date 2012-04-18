''''
-------------------------------------------------------------------------------
Filename   : ch1.py
Date       : 2012-04-16
Author     : C. Vogel
Purpose    : Replicate analysis of UFO data in Chapter 1 of _Machine Learning
           : for Hackers_.
Input Data : ufo_awesome.csv is available at the book's github repository at
           : https://github.com/johnmyleswhite/ML_for_Hackers.git
Libraries  : Numpy 1.6.1, Matplotlib 1.1.0, Pandas 0.7.3
-------------------------------------------------------------------------------

This script is a Python port of the R code in Chapter 1 of _Machine Learning
for Hackers_ by D. Conway and J.M. White. It is mainly intended to be run via
the interactive shell, though that's not necessary.

The script will produce (1) a cleaned tab-separated file of the UFO data;
(2) a series of 4 PNG figures.

The UFO dataset (approx. 75MB tab-separated file) should be located in a
/data/ufo subfolder of the working directory. Otherwise, change the `inpath`
and `outpath` variables at the start of the file.

For a detailed description of the analysis and the process of porting it
to Python, see: slendrmeans.wordpress.com/will-it-python.
'''


import numpy as np
from pandas import *
import matplotlib.pyplot as plt
import datetime as dt
import time
import re

# The location of the UFO raw data
inpath = 'data/ufo/ufo_awesome.tsv'

#########################################
# Fixing extra columns in the raw data. #
#########################################
# Pandas' read_table function gives an error reading the raw data file
# `ufo_awesome.tsv`. It turns out there are extra tabs in some of the fields,
# generating extra (>6) columns.

# A test: read lines from the file until we reach a line with more than 6
# tab-separated columns. Then print the line. I use enumerate() to identify
# the bad line and its columns.
# This 7th column of this bad line corresponds to the first bad date
# column in the text. (Indicating R pushes the extra columns to new lines
# to a new row).

inf = open(inpath, 'r')
for i, line in enumerate(inf):
    splitline = line.split('\t')
    if len(splitline) != 6:
        first_bad_line = splitline
        print "First bad row:", i
        for j, col in enumerate(first_bad_line):
            print j, col
        break
inf.close()

# The location of a cleaned version of the data, where the extra
# columns are eliminated. Output of the function `ufo_tab_to_sixcols` below.
outpath = 'data/ufo/ufo_awesome_6col.tsv'


def ufotab_to_sixcols(inpath, outpath):
    '''
    Keep only the first 6 columns of data from messy UFO TSV file.

    The UFO data set is only supposed to have six columns. But...

    The sixth column is a long written description of the UFO sighting, and
    sometimes is broken by tab characters which create extra columns.

    For these records, we only keep the first six columns. This typically cuts
    off some of the long description.

    Sometimes a line has less than six columns. These are not written to
    the output file (i.e., they're dropped from the data). These records are
    usually so comprimised as to be uncleanable anyway.

    This function has (is) a side effect on the `outpath` file, to which it
    writes output.
    '''

    inf = open(inpath, 'r')
    outf = open(outpath, 'w')

    for line in inf:
        splitline = line.split('\t')
        # Skip short lines, which are dirty beyond repair, anyway.
        if len(splitline) < 6:
            continue

        newline = ('\t').join(splitline[ :6])
        # Records that have been truncated won't end in a newline character
        # so add one.
        if newline[-1: ] != '\n':
            newline += '\n'       

        outf.write(newline)

    inf.close()
    outf.close()

# Run the data cleaning function to create the cleaned file. No need to do
# this more than once.
ufotab_to_sixcols(inpath, outpath)

# With the new clean file, we can use Pandas' to import the data.
ufo = read_table('data/ufo/ufo_awesome_6col.tsv', sep = '\t', na_values = '',
                 header = None, names = ['date_occurred',
                                         'date_reported',
                                         'location',
                                         'short_desc',
                                         'duration',
                                         'long_desc'])

# Print the beginning of the data; compare to table on p. 14.
print ufo.head(6).to_string(formatters = {'long_desc' : lambda x : x[ :21]})

#########################################
# Converting and cleaning up date data. #
#########################################
# Unlike the R import, Pandas' read_table pulled the dates in as integers
# in YYYYMMDD format. We'll use the function below and map() it to the
# date columns in the data.

def ymd_convert(x):
    '''
    Convert dates in the imported UFO data.
    Clean entries will look like YYYMMDD. If they're not clean, return NA.
    '''
    try:
        cnv_dt = dt.datetime.strptime(str(x), '%Y%m%d')
    except ValueError:
        cnv_dt = np.nan
        
    return cnv_dt

ufo['date_occurred'] = ufo['date_occurred'].map(ymd_convert)
ufo['date_reported'] = ufo['date_reported'].map(ymd_convert)

# Get rid of the rows that couldn't be conformed to datetime.
ufo = ufo[(notnull(ufo['date_reported'])) & (notnull(ufo['date_occurred']))]

#############################
# Organizing location data. #
#############################
# Note on p. 16 the authors claim strsplit() throws an error if there is no
# comma in the entry. This doesn't appear to be true.

def get_location(l):
    '''
    Divide the `location` variable in the data into two new variables.
    The first is the city, the second the state (or province). The function
    returns a two-element list of the form [city, state].

    This function is a fairly direct translation of the one in the text.
    But, by assuming legitimate U.S. locations have only one comma in them
    (e.g. `Baltimore, MD`), the authors miss a number of data points where
    the `city` entry has a detailed description with several commas: e.g.,
    `Baltimore, near U.S. Rte 59, MD`.
    '''
    split_location = l.split(',')
    clean_location = [x.strip() for x in split_location]
    if len(split_location) != 2:
        clean_location = ['', '']

    return clean_location

# As an alternative to the one-comma method for finding U.S. locations,
# we try using a regular expression that looks for entries that end in a
# comma and two letters (e.g., `, MD`) after stripping extra white space.

# Since the regexp is going to be mapped along a Series of data, we'll 
# compile it first.
us_state_pattern = re.compile(', [A-Z][A-Z]$', re.IGNORECASE)

def get_location2(l):
    '''
    Divide the `location` variable in the data into two new variables.
    The first is the city, the second the state (or province). The function
    returns a two-element list of the form [city, state].

    This function assumes legitimate U.S. locations have location data
    that end in a comma plus the two-letter state abbreviation. It will
    miss any rows where, for instance, the state is spelled out.

    Note that the regexp pattern `us_state_pattern` is defined outside
    the function, and not called as an extra argument. (Since this
    function will be used with Pandas' map(), it's more convenient to
    define it with a single argument.
    '''
    strip_location = l.strip()
    us_state_search = us_state_pattern.search(strip_location)
    if us_state_search == None:
        clean_location = ['', '']
    else: 
        us_city = strip_location[ :us_state_search.start()]
        us_state = strip_location[us_state_search.start() + 2: ]
        clean_location = [us_city, us_state]
    return clean_location
    
# Get a series of [city, state] lists, then unpack them into new
# variables in the data frame.
location_lists = ufo['location'].map(get_location2)
ufo['us_city'] = [city for city, st in location_lists]
ufo['us_state'] = [st.lower() for city, st in location_lists]

# State list from p. 18. Note they forget DC. There seem to be 12 DC entries.
us_states = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl',
             'ga', 'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma',
             'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne',
             'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri',
             'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv',
             'wy']



# If the `us_state` variable doesn't match the states in the list, set it to
# a missing string. Then nix the rows without valid U.S. states.
ufo['us_state'][-np.in1d(ufo['us_state'].tolist(), us_states)] = ''
ufo['us_city'][-np.in1d(ufo['us_state'].tolist(), us_states)] = ''

ufo_us = ufo[ufo['us_state'] != '']

# Get a data Series of years in the data. Note that Pandas' describe()
# won't give quantiles of the datetime objects in the date variables.
# (it requires interpolating between dates, which is tricky with
# datetimes. We can call describe() on the years to get quantiles, though
# since they're just integers.
years = ufo_us['date_occurred'].map(lambda x: x.year)
years.describe()


###########################################################################
# Plot distribution of sigthings over time and subset to recent sigthings #
###########################################################################
# Figure 1-5 of the text. Note it's over years, and not the original
# `date_occured` variable. Matplotlib apparently can't draw histograms
# of datetimes.
plt.figure()
years.hist(bins = (years.max() - years.min())/30., fc = 'steelblue')
plt.title('Histogram of years with U.S. UFO sightings\nAll years in data')
plt.savefig('quick_hist_all_years.png')

# Restrict the dates in the data to 1990 and after.
ufo_us = ufo_us[ufo_us['date_occurred'] >= dt.datetime(1990, 1, 1)]

years_post90 = ufo_us['date_occurred'].map(lambda x: x.year)

# How much data do we have now, compare to p. 22 of the text.
ufo_us.shape

# Check how many sightings we saved with the regex version of the
# location-cleaning function.
city_commas = ufo['us_city'].map(lambda x: x.count(','))
print 'Cities with commas = ', sum(city_commas > 0)

# Figure 1-6 in the text.
plt.figure()
years_post90.hist(bins = 20, fc = 'steelblue')
plt.title('Histogram of years with U.S. UFO sightings\n1990 through 2010')
plt.savefig('quick_hist_post90.png')

# It's a little strange to histogram over dates. Let's just make a line
# plot with the time series of no. of sigthings by date. Aggregated at the
# national level, it looks like there's some seasonality in the data,
# and a clear `millenium` effect.
post90_count = ufo_us.groupby('date_occurred')['date_occurred'].count()
plt.figure()
post90_count.plot()
plt.title('Number of U.S. UFO sightings\nJanuary 1990 through August 2010')
plt.savefig('post90_count_ts.png')

##################################
# Get monthly sightings by state #
##################################
# Aggregate data to the state/month level with Pandas' groupby() method.
ufo_us['year_month'] = ufo_us['date_occurred'].map(lambda x:
                                                   dt.date(x.year, x.month, 1))

sightings_counts = ufo_us.groupby(['us_state',
                                   'year_month'])['year_month'].count()

# Check out Alaska to compare with p. 22. Note we get an extra row, which
# results from the improved location cleaning.
print 'First few AK sightings in data:'
print sightings_counts.ix['ak'].head(6)

print 'Extra AK sighting, no on p. 22:'
print ufo_us[(ufo_us['us_state'] == 'ak') &
             (ufo_us['year_month'] == dt.date(1994, 2, 1))] \
             [['year_month','location']]

# Since groupby drops state-month levels for which there are no sightings,
# we'll create a 2-level MultiIndex with the full range of state-month pairs.
# Then, we'll re-index the data, filling in 0's where data is missing.
ym_list = [dt.date(y, m, 1) for y in range(1990, 2011)
                            for m in range(1, 13)
                            if dt.date(y, m, 1) <= dt.date(2010, 8, 1)]

full_index = zip(np.sort(us_states * len(ym_list)), ym_list * len(us_states))
full_index = MultiIndex.from_tuples(full_index, names =
                                    ['states', 'year_month'])

sightings_counts = sightings_counts.reindex(full_index, fill_value = 0)

##############################################################
# Plot monthly sightings by state in lattice/facet-wrap plot #
##############################################################
# Subplot parameters. We set up a figures with MxN subplots, where MxN >= 51
# (no. of states to plot). When MxN > 51, the `hangover` variable counts how
# many extra subplot remain in the last row of figure. We'll need this to
# to put tick labels in nice places.
nrow = 13; ncol = 4; hangover = len(us_states) % ncol

fig, axes = plt.subplots(nrow, ncol, sharey = True, figsize = (9, 11))

fig.suptitle('Monthly UFO Sightings by U.S. State\nJanuary 1990 through August 2010',
             size = 12)
plt.subplots_adjust(wspace = .05, hspace = .05)

num_state = 0
for i in range(nrow):
    for j in range(ncol):
        xs = axes[i, j]

        xs.grid(linestyle = '-', linewidth = .25, color = 'gray')

        if num_state < 51:
            st = us_states[num_state]
            sightings_counts.ix[st, ].plot(ax = xs, linewidth = .75)
            xs.text(0.05, .95, st.upper(), transform = axes[i, j].transAxes, 
                    verticalalignment = 'top')
            num_state += 1 
        else:
            # Make extra subplots invisible
            plt.setp(xs, visible = False)
            
        xtl = xs.get_xticklabels()
        ytl = xs.get_yticklabels()

        # X-axis tick labels:
        # Turn off tick labels for all the the bottom-most
        # subplots. This includes the plots on the last row, and
        # if the last row doesn't have a subplot in every column
        # put tick labels on the next row up for those last
        # columns.
        #
        # Y-axis tick labels:
        # Put left-axis labels on the first column of subplots,
        # odd rows. Put right-axis labels on the last column
        # of subplots, even rows.
        if i < nrow - 2 or (i < nrow - 1 and (hangover == 0 or
                            j <= hangover - 1)):
            plt.setp(xtl, visible = False)
        if j > 0 or i % 2 == 1:
            plt.setp(ytl, visible = False)
        if j == ncol - 1 and i % 2 == 1:
            xs.yaxis.tick_right()
          
        plt.setp(xtl, rotation=90.) 

plt.savefig('ufo_ts_bystate.png', dpi = 300)

