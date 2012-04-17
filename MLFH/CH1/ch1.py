import os
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
import datetime as dt
import time
import re

#os.chdir('/Users/cvogel/Documents/blog/MLFH/CH1/')

inpath = 'data/ufo/ufo_awesome.tsv'
outpath = 'data/ufo/ufo_awesome_6col.tsv'

###
inf = open(inpath, 'r')
for i, line in enumerate(inf):
    splitline = line.split('\t')
    if len(splitline) != 6:
        first_bad_line = splitline
        print i
        for j, col in enumerate(first_bad_line):
            print j, col
        break
inf.close()
###    
def ufotab_to_sixcols(inpath, outpath):
    '''
    Keep only the first 6 columns of data from messy UFO TSV file.

    The UFO data set is only supposed to have six columns. But...

    Sometimes the last column is a long written description of the UFO
    sighting, and sometimes is broken by tab characters which create extra
    columns.

    For these records, we only keep the first six columns. This typically cuts
    off some of the long description.

    And sometimes a line has less than six columns. These are not written to
    the output file (i.e., they're dropped from the data). These records are
    usually so comprimised as to be uncleanable anyway.

    This function has (is) a side effect on the `outpath` file, to which it
    writes output. These lines are typically so comprimised as to be unclean-
    able.
    '''

    inf = open(inpath, 'r')
    outf = open(outpath, 'w')

    for i,line in enumerate(inf):
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

ufo = read_table('data/ufo/ufo_awesome_6col.tsv', sep = '\t', na_values = '',
                 header = None, names = ['date_occurred',
                                         'date_reported',
                                         'location',
                                         'short_desc',
                                         'duration',
                                         'long_desc'])

#print ufo.head(6).to_string

# Converting date strings and dealing with mal-formed data.
# Unlike the R import, Pandas read_table pulled these in as numbers.

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


# Organizing location data.
# Note on p. 16 they claim strsplit throws an error if there is no comma.
# This doesn't appear to be true.

def get_location(l):
    split_location = l.split(',')
    clean_location = [x.strip() for x in split_location]
    if len(split_location) != 2:
        clean_location = ['', '']

    return clean_location

us_state_pattern = re.compile(', [A-Z][A-Z]$', re.IGNORECASE)

def get_location2(l):
    strip_location = l.strip()
    us_state_search = us_state_pattern.search(strip_location)
    if us_state_search == None:
        clean_location = ['', '']
    else: 
        us_city = strip_location[ :us_state_search.start()]
        us_state = strip_location[us_state_search.start() + 2: ]
        clean_location = [us_city, us_state]
    return clean_location
    

location_lists = ufo['location'].map(get_location2)
ufo['us_city'] = [city for city, st in location_lists]
ufo['us_state'] = [st.lower() for city, st in location_lists]

# State list from p. 18. Note they forget DC. There seem to be 12 DC entries.
# Had to use groupby to work this out. ufo['us_state'] == 'DC' threw nans.
# This is a recognized bug fixed in 0.7.3.
us_states = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl',
             'ga', 'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma',
             'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne',
             'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri',
             'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv',
             'wy']

ufo['us_state'][-np.in1d(ufo['us_state'].tolist(), us_states)] = ''
ufo['us_city'][-np.in1d(ufo['us_state'].tolist(), us_states)] = ''

ufo_us = ufo[ufo['us_state'] != '']
             
years = ufo_us['date_occurred'].map(lambda x: x.year)
years.describe()

plt.figure()
years.hist(bins = (years.max() - years.min())/30., fc = 'steelblue')
plt.title('Histogram of years with U.S. UFO sightings\nAll years in data')
plt.savefig('quick_hist_all_years.png')

ufo_us = ufo_us[ufo_us['date_occurred'] >= dt.datetime(1990, 1, 1)]

years_post90 = ufo_us['date_occurred'].map(lambda x: x.year)

ufo_us.shape

# Check how many sightings we saved with the regex.
city_commas = ufo['us_city'].map(lambda x: x.count(','))
print 'Cities with commas = %i' % sum(city_commas > 0)

plt.figure()
years_post90.hist(bins = 20, fc = 'steelblue')
plt.title('Histogram of years with U.S. UFO sightings\n1990 through 2010')
plt.savefig('quick_hist_post90.png')

post90_count = ufo_us.groupby('date_occurred')['date_occurred'].count()
plt.figure()
post90_count.plot()
plt.title('Number of U.S. UFO sightings\nJanuary 1990 through August 2010')
plt.savefig('post90_count_ts.png')

ufo_us['year_month'] = ufo_us['date_occurred'].map(lambda x:
                                                   dt.date(x.year, x.month, 1))

sightings_counts = ufo_us.groupby(['us_state',
                                   'year_month'])['year_month'].count()
sightings_counts.ix['ak'].head(6)

ufo_us[(ufo_us['us_state'] == 'ak') &
       (ufo_us['year_month'] == dt.date(1994, 2, 1))]['location']

ym_list = [dt.date(y, m, 1) for y in range(1990, 2011)
                            for m in range(1, 13)
                            if dt.date(y, m, 1) <= dt.date(2010, 8, 1)]

full_index = zip(np.sort(us_states * len(ym_list)), ym_list * len(us_states))
full_index = MultiIndex.from_tuples(full_index, names =
                                    ['states', 'year_month'])

sightings_counts = sightings_counts.reindex(full_index, fill_value = 0)

nrow = 13; ncol = 4; hangover = len(us_states) % ncol

fig, axes = plt.subplots(nrow, ncol, sharey = True, figsize = (9, 11))

fig.patch.set_fc('white')
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
            plt.setp(xs, visible = False)
            
        xtl = xs.get_xticklabels()
        ytl = xs.get_yticklabels()
        
        if i < nrow - 2 or (i < nrow - 1 and (hangover == 0 or
                            j <= hangover - 1)):
            plt.setp(xtl, visible = False)
        if j > 0 or i % 2 == 1:
            plt.setp(ytl, visible = False)
        if j == ncol - 1 and i % 2 == 1:
            xs.yaxis.tick_right()
          
        plt.setp(xtl, rotation=90.) 


plt.savefig('ufo_ts_bystate.png', dpi = 300)

