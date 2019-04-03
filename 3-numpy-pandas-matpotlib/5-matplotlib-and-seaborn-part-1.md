# 1 - Instructor
Mike Yi
  - Ph.D in Cognitive Science from UC Irvine
matplotlib & seaborn are useful for data visualization

# 2 - Introduction
Univariate Visualizations
  - visualizations of single variables
Clean data
  - look for oddities, missing values, etc
Chart types
  - bar charts for qualitative variables
  - histograms for quantitative variables

# 3 - Tidy Data
Tidy Dataset
  - tabular dataset
  - each variable is a column
  - each observation is a row
  - each type of observational unit is a table
This course will provide us with tidy data but in reality we may need to clean up our data before we begin

# 4 - Bar Charts
Depicts distribution of a categorical variable
  - each level of cat var is depicted with a bar
  - height indicates frequency of data points on that level

`sb.countplot(data = df, x = 'cat_var')`

If it's better to have all bars the same color:

```
base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'cat_var', color = base_color)
```

Common to sort the data in terms of frequency:

```
base_color = sb.color_palette()[0]
cat_order = df['cat_var'].value_counts().index
sb.countplot(data = df, x = 'cat_var', color = base_color, order = cat_order)
```

Create an order that the bars should be sorted in:

```
# this method requires pandas v0.21 or later
level_order = ['Alpha', 'Beta', 'Gamma', 'Delta']
ordered_cat = pd.api.types.CategoricalDtype(ordered = True, categories = level_order)
df['cat_var'] = df['cat_var'].astype(ordered_cat)

# # use this method if you have pandas v0.20.3 or earlier
# df['cat_var'] = df['cat_var'].astype('category', ordered = True,
#                                      categories = level_order)

base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'cat_var', color = base_color)
```

Additional Variations
  - if data is in pandas series, 1d NumPy array or list, you can set it as first argument to `countplot` function
  - Make a horizontal bar chart as follows
  ```
  base_color = sb.color_palette()[0]
  sb.countplot(data = df, y = 'cat_var', color = base_color)
  ```

  Could also use matplotlibs `xticks` function's "rotation" param to change orientation of labels

  ```
  base_color = sb.color_palette()[0]
  sb.countplot(data = df, x = 'cat_var', color = base_color)
  plt.xticks(rotation = 90)
  ```

# 5 - Absolute vs. Relative Frequency
Default for seaborn is absolute frequency

You might be interested in which proportion of the data falls in a given category
  - can change the chart's axis to relative frequency

```
# get proportion taken by most common group for derivation
# of tick marks
n_points = df.shape[0]
max_count = df['cat_var'].value_counts().max()
max_prop = max_count / n_points

# generate tick mark locations and names
tick_props = np.arange(0, max_prop, 0.05)
tick_names = ['{:0.2f}'.format(v) for v in tick_props]

# create the plot
base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'cat_var', color = base_color)
plt.yticks(tick_props * n_points, tick_names)
plt.ylabel('proportion')
```

You could also use `text annotations` to label frequencies on the bars instead

```
# create the plot
base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'cat_var', color = base_color)

# add annotations
n_points = df.shape[0]
cat_counts = df['cat_var'].value_counts()
locs, labels = plt.xticks() # get the current tick locations and labels

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = cat_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/n_points)

    # print the annotation just below the top of the bar
    plt.text(loc, count-8, pct_string, ha = 'center', color = 'w')
```

# 6 - Counting Mising Data
Apply bar charts to visualize missing data
  - create table with number of missing values in each column
  - `df.isna().sum()`

```
na_counts = df.isna().sum()
base_color = sb.color_palette()[0]
sb.barplot(na_counts.index.values, na_counts, color = base_color)
```
  - this is handy to make bar charts for summarized data
  - if data is not summarized, used previously taught `countplot` to avoid having to summarize data first

# 7 - Bar Chart Practice
Practiced making a couple of bar charts
  - horizontal and vertical
  - absolute and relative frequency
  - sorted and unsorted

# 8 - Pie Charts
Depicts relative frequencies for levels of categorical variable
  - donut is similar but has center removed

When to use a pie chart:
  - must be interested in relative frequencies
  - want to show how whole is broken into parts
  - works best with only 2-3 slices
  - plot data systemically (usually clockwise from top - most to least frequent)

Bar chart is always a good alternative to a pie chart

```
# code for the pie chart seen above
sorted_counts = df['cat_var'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False);
plt.axis('square')
```
  - `startangle = 90`: starts first slice vertically upward
  - `counterclock = False` plots sorted counts in clockwise fashion
  - `plt.axis('square')` makes scaling of plot equal on x and y axes

Create a donut chart as follows:
```
sorted_counts = df['cat_var'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False, wedgeprops = {'width' : 0.4});
plt.axis('square')
```
  - by default the `wedgeprops` width is 1
  - make it less than one and remove color from the center of the circle

# 9 - Histograms
Plot distribution of numeric value
Quantitative version of bar chart
Values are grouped into contious bins

```
plt.hist(data = df, x = 'num_var')
```

By default, `hist` divides data into 10 bins
  - you will almost always want to adjust this
  ```
  bin_edges = np.arange(0, df['num_var'].max()+1, 1)
  plt.hist(data = df, x = 'num_var', bins = bin_edges)
  ```
  - `arange` arguments:
    - leftmost bin edge
    - upper limit
    - bin width
  * Play around with different bin sizes to see what works best

Can also use seaborn's `distplot`
  - `sb.distplot(df['num_var'])`
  - first argument must be series or array with points to be plotted
  - better bin defaults but still want to mess with them
  ```
  bin_edges = np.arange(0, df['num_var'].max()+1, 1)
  sb.distplot(df['num_var'], bins = bin_edges, kde = False,
              hist_kws = {'alpha' : 1})
  ```

# 10 - Histogram Practice
Made a histogram. It was fun. But that is a lie.

# 11 - Figures, Axes, and Subplots
Base of visulaization in matplotlib is a `Figure` object
  - will have one or more `Axes` objects

```
fig = plt.figure()
ax = fig.add_axes([.125, .125, .775, .755])
ax.hist(data = df, x = 'num_var')
```
  - `figure()` creates new Figure object
  - `.add_axes()` creates Axes object in the figure (provide dimensions, width, height)
  - `.hist()` says which type of graph to plot on there

In seaborn, seaborn functions have an `ax` parm to specify which Axes a plot will be drawn on

```
fig = plt.figure()
ax = fig.add_axes([.125, .125, .775, .755])
base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'cat_var', color = base_color, ax = ax)
```

To get two or more graphs side by side, you can use subplots!

```
plt.figure(figsize = [10, 5]) # larger figure size for subplots

# example of somewhat too-large bin size
plt.subplot(1, 2, 1) # 1 row, 2 cols, subplot 1
plt.hist(pokemon['special-defense'])

# example of somewhat too-small bin size
plt.subplot(1, 2, 2) # 1 row, 2 cols, subplot 1
plt.hist(pokemon['special-defense'], bins = 80)
```

# 12 - Choosing a Plot for Discrete Data
You can use `rwidth` to separate bars on a histogram

# 13 - Descriptive Statistics, Outliers and Axis Limits
Pay attention to limits, scale, etc to find underlying patterns in data

Change histograms axis limits you can add `xlim` to your code
  - takes two numbers specifying upper and lower bounds of x-axis)
  - allows you to zoom in on more interesting data without focusing on outliers

# 14 - Scales and Tranformations
Sometimes different scales (e.g. log-normals) are better for showcasing the data

```
plt.figure(figsize = [10, 5])

# left histogram: data plotted in natural units
plt.subplot(1, 2, 1)
bin_edges = np.arange(0, data.max()+100, 100)
plt.hist(data, bins = bin_edges)
plt.xlabel('values')

# right histogram: data plotted after direct log transformation
plt.subplot(1, 2, 2)
log_data = np.log10(data) # direct data transform
log_bin_edges = np.arange(0.8, log_data.max()+0.1, 0.1)
plt.hist(log_data, bins = log_bin_edges)
plt.xlabel('log(values)')
```

Can also do a scale transformation
```
bin_edges = 10 ** np.arange(0.8, np.log10(data.max())+0.1, 0.1)
plt.hist(data, bins = bin_edges)
plt.xscale('log')
tick_locs = [10, 30, 100, 300, 1000, 3000]
plt.xticks(tick_locs, tick_locs)
```
  - make sure to specify `xticks` after `xscale`

# 15 - Scales and Transformations Practice
Practiced zooming, scaling and transforming charts.

# 16 - Lesson Summary
Good summary of what we've learned.
  - my brain is fried
  - use the right bar chart or histogram for the job
  - don't use pie charts because no one can read them
  - lots of info about how to plot things....you can read the documentation as you need it

# 17 - Extra: Kernel Density Estimation
  - details about how the kernel density estimation line works in Seaborn
  - if you need to learn how to interpret it, look [here](https://classroom.udacity.com/nanodegrees/nd089/parts/8de94dee-7635-43b3-9d11-5e4583f22ce3/modules/dd3e4af8-d576-427e-baae-925fd16ff2ff/lessons/43855abb-dae5-46c7-9167-4b001dabd41e/concepts/cfbaee5f-1a27-4457-8a67-8421bf19cb05)

```
data = [0.0, 3.0, 4.5, 8.0]
plt.figure(figsize = [12, 5])

# left plot: showing kde lumps with the default settings
plt.subplot(1, 3, 1)
sb.distplot(data, hist = False, rug = True, rug_kws = {'color' : 'r'})

# central plot: kde with narrow bandwidth to show individual probability lumps
plt.subplot(1, 3, 2)
sb.distplot(data, hist = False, rug = True, rug_kws = {'color' : 'r'},
            kde_kws = {'bw' : 1})

# right plot: choosing a different, triangular kernel function (lump shape)
plt.subplot(1, 3, 3)
sb.distplot(data, hist = False, rug = True, rug_kws = {'color' : 'r'},
            kde_kws = {'bw' : 1.5, 'kernel' : 'tri'})
```
