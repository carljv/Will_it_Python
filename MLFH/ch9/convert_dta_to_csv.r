# Convert Senate data Stata files to CSV
# For Machine Learning for Hackers Chapter 9

wkdir = "data/roll_call"
setwd(wkdir)
library(foreign)

# dta files are in 'data/roll_call'
# csv files will be written to the same directory.
flist = list.files()

for (f in flist) {

    # Create filename xyz123.csv from xyz123.dta
    csv_name = paste(strsplit(f, "\\.")[[1]][1], "csv", sep = ".")

    df = read.dta(f)
    write.csv(df, csv_name)
}